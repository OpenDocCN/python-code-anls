# `D:\src\scipysrc\pandas\pandas\tests\reshape\test_get_dummies.py`

```
# 导入正则表达式模块
import re
# 导入Unicode数据模块，用于处理Unicode字符
import unicodedata

# 导入NumPy库并使用np作为别名
import numpy as np
# 导入pytest测试框架
import pytest

# 导入Pandas库的测试装饰器模块
import pandas.util._test_decorators as td

# 从Pandas核心数据类型中导入is_integer_dtype函数
from pandas.core.dtypes.common import is_integer_dtype

# 导入Pandas库并使用pd作为别名
import pandas as pd
# 从Pandas库中导入特定对象
from pandas import (
    ArrowDtype,
    Categorical,
    CategoricalDtype,
    CategoricalIndex,
    DataFrame,
    Index,
    RangeIndex,
    Series,
    SparseDtype,
    get_dummies,
)
# 导入Pandas测试工具模块并使用tm作为别名
import pandas._testing as tm

# 从Pandas稀疏数组模块中导入SparseArray对象
from pandas.core.arrays.sparse import SparseArray

# 尝试导入pyarrow库，如果失败则将pa设为None
try:
    import pyarrow as pa
except ImportError:
    pa = None

# 定义测试类TestGetDummies
class TestGetDummies:
    # 定义测试用的数据帧df作为fixture
    @pytest.fixture
    def df(self):
        return DataFrame({"A": ["a", "b", "a"], "B": ["b", "b", "c"], "C": [1, 2, 3]})

    # 定义测试用的数据类型dtype作为fixture，支持多种数据类型
    @pytest.fixture(params=["uint8", "i8", np.float64, bool, None])
    def dtype(self, request):
        return np.dtype(request.param)

    # 定义sparse参数作为fixture，用于测试稀疏和密集数据
    @pytest.fixture(params=["dense", "sparse"])
    def sparse(self, request):
        # 参数是字符串以简化读取测试结果，如TestGetDummies::test_basic[uint8-sparse]
        return request.param == "sparse"

    # 返回有效的数据类型，如果为None则返回np.uint8
    def effective_dtype(self, dtype):
        if dtype is None:
            return np.uint8
        return dtype

    # 测试当数据帧包含dtype为object类型时，get_dummies是否会引发ValueError异常
    def test_get_dummies_raises_on_dtype_object(self, df):
        msg = "dtype=object is not a valid dtype for get_dummies"
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, dtype="object")

    # 测试get_dummies函数的基本功能，包括稀疏和密集数据及不同数据类型的处理
    def test_get_dummies_basic(self, sparse, dtype):
        s_list = list("abc")
        s_series = Series(s_list)
        s_series_index = Series(s_list, list("ABC"))

        # 期望的结果DataFrame
        expected = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
            dtype=self.effective_dtype(dtype),
        )

        # 如果sparse为True，则使用SparseArray处理期望的结果
        if sparse:
            if dtype.kind == "b":
                expected = expected.apply(SparseArray, fill_value=False)
            else:
                expected = expected.apply(SparseArray, fill_value=0.0)

        # 测试处理列表数据的结果
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        # 测试处理Series对象的结果
        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        # 将期望的DataFrame索引设置为指定列表，并测试处理Series索引的结果
        expected.index = list("ABC")
        result = get_dummies(s_series_index, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试 get_dummies 函数处理基本类型的情况
    def test_get_dummies_basic_types(self, sparse, dtype, using_infer_string):
        # GH 10531
        # 创建一个包含字符 'abc' 的列表 s_list
        s_list = list("abc")
        # 根据 s_list 创建 Series 对象 s_series
        s_series = Series(s_list)
        # 创建 DataFrame 对象 s_df，包含三列数据
        s_df = DataFrame(
            {"a": [0, 1, 0, 1, 2], "b": ["A", "A", "B", "C", "C"], "c": [2, 3, 3, 3, 2]}
        )

        # 创建期望的 DataFrame 对象 expected，包含三列数据
        expected = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]},
            dtype=self.effective_dtype(dtype),  # 指定数据类型
            columns=list("abc"),  # 指定列名
        )

        # 如果 sparse=True，则根据 dtype 类型选择填充值
        if sparse:
            if is_integer_dtype(dtype):
                fill_value = 0
            elif dtype == bool:
                fill_value = False
            else:
                fill_value = 0.0

            # 将 expected 应用 SparseArray 处理
            expected = expected.apply(SparseArray, fill_value=fill_value)

        # 使用 get_dummies 处理 s_list，并验证结果与 expected 是否相等
        result = get_dummies(s_list, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        # 使用 get_dummies 处理 s_series，并验证结果与 expected 是否相等
        result = get_dummies(s_series, sparse=sparse, dtype=dtype)
        tm.assert_frame_equal(result, expected)

        # 使用 get_dummies 处理 s_df，并验证结果与 expected 是否相等
        result = get_dummies(s_df, columns=s_df.columns, sparse=sparse, dtype=dtype)

        # 如果 sparse=True，则设置 dtype_name 为 Sparse[类型名称, 填充值]
        if sparse:
            dtype_name = f"Sparse[{self.effective_dtype(dtype).name}, {fill_value}]"
        else:
            dtype_name = self.effective_dtype(dtype).name

        # 创建期望的 Series 对象 expected，包含 count 字段
        expected = Series({dtype_name: 8}, name="count")
        # 获取 result 的数据类型计数，转换索引为字符串，验证结果与 expected 是否相等
        result = result.dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        tm.assert_series_equal(result, expected)

        # 使用 get_dummies 处理 s_df 的列 'a'，并验证结果与 expected 是否相等
        result = get_dummies(s_df, columns=["a"], sparse=sparse, dtype=dtype)

        # 根据 using_infer_string 确定 key 的值为 'string' 或 'object'
        key = "string" if using_infer_string else "object"
        # 创建期望的类型计数字典 expected_counts
        expected_counts = {"int64": 1, key: 1}
        expected_counts[dtype_name] = 3 + expected_counts.get(dtype_name, 0)

        # 创建期望的 Series 对象 expected，包含 count 字段，按索引排序
        expected = Series(expected_counts, name="count").sort_index()
        # 获取 result 的数据类型计数，转换索引为字符串，按索引排序，验证结果与 expected 是否相等
        result = result.dtypes.value_counts()
        result.index = [str(i) for i in result.index]
        result = result.sort_index()
        tm.assert_series_equal(result, expected)

    # 定义测试函数，用于测试 get_dummies 函数处理只含 NaN 的情况
    def test_get_dummies_just_na(self, sparse):
        # 创建只含一个 NaN 的列表 just_na_list 和相应的 Series 对象
        just_na_list = [np.nan]
        just_na_series = Series(just_na_list)
        just_na_series_index = Series(just_na_list, index=["A"])

        # 使用 get_dummies 处理 just_na_list，验证结果为空
        res_list = get_dummies(just_na_list, sparse=sparse)
        assert res_list.empty

        # 使用 get_dummies 处理 just_na_series，验证结果为空
        res_series = get_dummies(just_na_series, sparse=sparse)
        assert res_series.empty

        # 使用 get_dummies 处理 just_na_series_index，验证结果为空
        res_series_index = get_dummies(just_na_series_index, sparse=sparse)
        assert res_series_index.empty

        # 验证 res_list 的索引列表为 [0]
        assert res_list.index.tolist() == [0]
        # 验证 res_series 的索引列表为 [0]
        assert res_series.index.tolist() == [0]
        # 验证 res_series_index 的索引列表为 ["A"]
        assert res_series_index.index.tolist() == ["A"]
    # 定义一个测试函数，用于测试 get_dummies 函数对包含 NaN 值的情况处理是否正确
    def test_get_dummies_include_na(self, sparse, dtype):
        # 创建一个包含字符串和 NaN 值的列表
        s = ["a", "b", np.nan]
        # 调用 get_dummies 函数，生成结果 DataFrame
        res = get_dummies(s, sparse=sparse, dtype=dtype)
        # 创建期望的 DataFrame，其中列名是字符串 'a' 和 'b'，每列对应的值表示是否包含对应的值
        exp = DataFrame(
            {"a": [1, 0, 0], "b": [0, 1, 0]}, dtype=self.effective_dtype(dtype)
        )
        # 如果 sparse 参数为 True，则根据 dtype 的种类设置期望的 DataFrame 为 SparseArray 格式
        if sparse:
            if dtype.kind == "b":
                exp = exp.apply(SparseArray, fill_value=False)
            else:
                exp = exp.apply(SparseArray, fill_value=0.0)
        # 使用 assert_frame_equal 函数比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(res, exp)

        # 测试包含 NaN 值时，dummy_na 参数为 True 的情况
        res_na = get_dummies(s, dummy_na=True, sparse=sparse, dtype=dtype)
        # 创建包含 NaN 值的期望 DataFrame
        exp_na = DataFrame(
            {np.nan: [0, 0, 1], "a": [1, 0, 0], "b": [0, 1, 0]},
            dtype=self.effective_dtype(dtype),
        )
        # 重新索引期望的 DataFrame，确保列的顺序正确
        exp_na = exp_na.reindex(["a", "b", np.nan], axis=1)
        # hack (NaN handling in assert_index_equal)
        exp_na.columns = res_na.columns
        # 如果 sparse 参数为 True，则根据 dtype 的种类设置期望的 DataFrame 为 SparseArray 格式
        if sparse:
            if dtype.kind == "b":
                exp_na = exp_na.apply(SparseArray, fill_value=False)
            else:
                exp_na = exp_na.apply(SparseArray, fill_value=0.0)
        # 使用 assert_frame_equal 函数比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(res_na, exp_na)

        # 测试只包含 NaN 值的情况，dummy_na 参数为 True 的情况
        res_just_na = get_dummies([np.nan], dummy_na=True, sparse=sparse, dtype=dtype)
        # 创建只包含 NaN 值的期望 DataFrame
        exp_just_na = DataFrame(
            Series(1, index=[0]), columns=[np.nan], dtype=self.effective_dtype(dtype)
        )
        # 使用 assert_numpy_array_equal 函数比较结果和期望的值是否相等
        tm.assert_numpy_array_equal(res_just_na.values, exp_just_na.values)

    # 定义一个测试函数，用于测试 get_dummies 函数对 Unicode 字符串的处理是否正确
    def test_get_dummies_unicode(self, sparse):
        # Unicode 字符串示例
        e = "e"
        eacute = unicodedata.lookup("LATIN SMALL LETTER E WITH ACUTE")
        s = [e, eacute, eacute]
        # 调用 get_dummies 函数，生成结果 DataFrame
        res = get_dummies(s, prefix="letter", sparse=sparse)
        # 创建期望的 DataFrame，其中列名分别是 'letter_e' 和 'letter_é'，每列对应的值表示是否包含对应的值
        exp = DataFrame(
            {"letter_e": [True, False, False], f"letter_{eacute}": [False, True, True]}
        )
        # 如果 sparse 参数为 True，则将期望的 DataFrame 转换为 SparseArray 格式
        if sparse:
            exp = exp.apply(SparseArray, fill_value=False)
        # 使用 assert_frame_equal 函数比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(res, exp)

    # 定义一个测试函数，用于测试 get_dummies 函数对 DataFrame 所有列为对象类型时的处理是否正确
    def test_dataframe_dummies_all_obj(self, df, sparse):
        # 从 DataFrame 中选择列 'A' 和 'B'
        df = df[["A", "B"]]
        # 调用 get_dummies 函数，生成结果 DataFrame
        result = get_dummies(df, sparse=sparse)
        # 创建期望的 DataFrame，其中列名包含 'A_a', 'A_b', 'B_b', 'B_c'，每列对应的值表示是否包含对应的值
        expected = DataFrame(
            {"A_a": [1, 0, 1], "A_b": [0, 1, 0], "B_b": [1, 1, 0], "B_c": [0, 0, 1]},
            dtype=bool,
        )
        # 如果 sparse 参数为 True，则将期望的 DataFrame 转换为 SparseArray 格式
        if sparse:
            expected = DataFrame(
                {
                    "A_a": SparseArray([1, 0, 1], dtype="bool"),
                    "A_b": SparseArray([0, 1, 0], dtype="bool"),
                    "B_b": SparseArray([1, 1, 0], dtype="bool"),
                    "B_c": SparseArray([0, 0, 1], dtype="bool"),
                }
            )
        # 使用 assert_frame_equal 函数比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    # 测试函数：测试处理包含字符串数据类型的DataFrame，生成虚拟变量
    def test_dataframe_dummies_string_dtype(self, df, using_infer_string):
        # 从原始DataFrame中选择列"A"和"B"
        df = df[["A", "B"]]
        # 将列"A"转换为Python对象类型，列"B"转换为字符串类型
        df = df.astype({"A": "object", "B": "string"})
        # 使用get_dummies函数生成虚拟变量
        result = get_dummies(df)
        # 期望的DataFrame结果，包含布尔类型的数据
        expected = DataFrame(
            {
                "A_a": [1, 0, 1],
                "A_b": [0, 1, 0],
                "B_b": [1, 1, 0],
                "B_c": [0, 0, 1],
            },
            dtype=bool,
        )
        # 如果不使用infer_string，则将"B_b"和"B_c"列转换为布尔类型
        if not using_infer_string:
            expected[["B_b", "B_c"]] = expected[["B_b", "B_c"]].astype("boolean")
        # 使用assert_frame_equal函数断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试处理包含不同数据类型的DataFrame，生成虚拟变量
    def test_dataframe_dummies_mix_default(self, df, sparse, dtype):
        # 使用get_dummies函数生成虚拟变量，根据sparse和dtype参数选择处理方式
        result = get_dummies(df, sparse=sparse, dtype=dtype)
        # 如果sparse为True，使用SparseArray；如果dtype的kind为"b"，使用SparseDtype(dtype, False)，否则使用SparseDtype(dtype, 0)
        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype
        # 期望的DataFrame结果，包含"C", "A_a", "A_b", "B_b", "B_c"列
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A_a": arr([1, 0, 1], dtype=typ),
                "A_b": arr([0, 1, 0], dtype=typ),
                "B_b": arr([1, 1, 0], dtype=typ),
                "B_c": arr([0, 0, 1], dtype=typ),
            }
        )
        # 从期望结果中选择"C", "A_a", "A_b", "B_b", "B_c"列
        expected = expected[["C", "A_a", "A_b", "B_b", "B_c"]]
        # 使用assert_frame_equal函数断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试处理DataFrame并指定前缀列表生成虚拟变量
    def test_dataframe_dummies_prefix_list(self, df, sparse):
        # 定义前缀列表，生成虚拟变量，并根据sparse参数选择处理方式
        prefixes = ["from_A", "from_B"]
        result = get_dummies(df, prefix=prefixes, sparse=sparse)
        # 期望的DataFrame结果，包含"C", "from_A_a", "from_A_b", "from_B_b", "from_B_c"列
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "from_A_a": [True, False, True],
                "from_A_b": [False, True, False],
                "from_B_b": [True, True, False],
                "from_B_c": [False, False, True],
            },
        )
        # 将期望结果的"C"列替换为原始DataFrame的"C"列
        expected[["C"]] = df[["C"]]
        # 定义列名列表，选择"from_A_a", "from_A_b", "from_B_b", "from_B_c"列
        cols = ["from_A_a", "from_A_b", "from_B_b", "from_B_c"]
        # 将"from_A_a", "from_A_b", "from_B_b", "from_B_c"列的数据类型应用到期望结果中
        typ = SparseArray if sparse else Series
        expected[cols] = expected[cols].apply(lambda x: typ(x))
        # 使用assert_frame_equal函数断言结果与期望是否相等
        tm.assert_frame_equal(result, expected)
    # 测试函数：使用指定前缀生成虚拟变量，处理DataFrame，并进行比较
    def test_dataframe_dummies_prefix_str(self, df, sparse):
        # 调用函数生成带有指定前缀的虚拟变量DataFrame
        result = get_dummies(df, prefix="bad", sparse=sparse)
        # 定义预期的列名列表，注意包含重复列名
        bad_columns = ["bad_a", "bad_b", "bad_b", "bad_c"]
        # 构建预期的DataFrame
        expected = DataFrame(
            [
                [1, True, False, True, False],
                [2, False, True, True, False],
                [3, True, False, False, True],
            ],
            columns=["C"] + bad_columns,  # 指定列名
        )
        # 将预期DataFrame中的数值类型转换为int64
        expected = expected.astype({"C": np.int64})
        if sparse:
            # 在处理稀疏数据时，通过pd.concat进行列的处理和类型分配
            # 避免列名重复的问题，参考GitHub上的问题解决方案
            expected = pd.concat(
                [
                    Series([1, 2, 3], name="C"),  # 列'C'的数据
                    Series([True, False, True], name="bad_a", dtype="Sparse[bool]"),
                    Series([False, True, False], name="bad_b", dtype="Sparse[bool]"),
                    Series([True, True, False], name="bad_b", dtype="Sparse[bool]"),
                    Series([False, False, True], name="bad_c", dtype="Sparse[bool]"),
                ],
                axis=1,
            )

        # 使用测试框架断言生成的结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试函数：使用子集列生成虚拟变量，处理DataFrame，并进行比较
    def test_dataframe_dummies_subset(self, df, sparse):
        # 调用函数生成带有指定前缀和列的虚拟变量DataFrame
        result = get_dummies(df, prefix=["from_A"], columns=["A"], sparse=sparse)
        # 构建预期的DataFrame
        expected = DataFrame(
            {
                "B": ["b", "b", "c"],  # 列'B'的数据
                "C": [1, 2, 3],  # 列'C'的数据
                "from_A_a": [1, 0, 1],  # 列'from_A_a'的
    # 测试函数：验证当 'prefix' 参数长度与待编码列长度不匹配时，是否引发 ValueError 异常，并验证异常消息是否与预期相符
    def test_dataframe_dummies_prefix_bad_length(self, df, sparse):
        msg = re.escape(
            "Length of 'prefix' (1) did not match the length of the columns being "
            "encoded (2)"
        )
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, prefix=["too few"], sparse=sparse)

    # 测试函数：验证当 'prefix_sep' 参数长度与待编码列长度不匹配时，是否引发 ValueError 异常，并验证异常消息是否与预期相符
    def test_dataframe_dummies_prefix_sep_bad_length(self, df, sparse):
        msg = re.escape(
            "Length of 'prefix_sep' (1) did not match the length of the columns being "
            "encoded (2)"
        )
        with pytest.raises(ValueError, match=msg):
            get_dummies(df, prefix_sep=["bad"], sparse=sparse)

    # 测试函数：使用自定义的前缀字典对 DataFrame 进行独热编码，验证编码结果是否符合预期
    def test_dataframe_dummies_prefix_dict(self, sparse):
        # 定义前缀字典
        prefixes = {"A": "from_A", "B": "from_B"}
        # 创建测试用 DataFrame
        df = DataFrame({"C": [1, 2, 3], "A": ["a", "b", "a"], "B": ["b", "b", "c"]})
        # 执行独热编码
        result = get_dummies(df, prefix=prefixes, sparse=sparse)

        # 预期的编码结果 DataFrame
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "from_A_a": [1, 0, 1],
                "from_A_b": [0, 1, 0],
                "from_B_b": [1, 1, 0],
                "from_B_c": [0, 0, 1],
            }
        )

        # 将特定列转换为布尔类型，如果使用稀疏格式则转换为稀疏布尔类型
        columns = ["from_A_a", "from_A_b", "from_B_b", "from_B_c"]
        expected[columns] = expected[columns].astype(bool)
        if sparse:
            expected[columns] = expected[columns].astype(SparseDtype("bool", False))

        # 使用测试框架验证实际结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

    # 测试函数：对包含缺失值的 DataFrame 执行独热编码，验证编码结果是否符合预期
    def test_dataframe_dummies_with_na(self, df, sparse, dtype):
        # 在 DataFrame 中添加缺失值
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        # 执行带有缺失值处理的独热编码
        result = get_dummies(df, dummy_na=True, sparse=sparse, dtype=dtype).sort_index(
            axis=1
        )

        # 根据是否使用稀疏格式，选择相应的数组类型和数据类型
        if sparse:
            arr = SparseArray
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            arr = np.array
            typ = dtype

        # 预期的编码结果 DataFrame，包含处理后的缺失值列
        expected = DataFrame(
            {
                "C": [1, 2, 3, np.nan],
                "A_a": arr([1, 0, 1, 0], dtype=typ),
                "A_b": arr([0, 1, 0, 0], dtype=typ),
                "A_nan": arr([0, 0, 0, 1], dtype=typ),
                "B_b": arr([1, 1, 0, 0], dtype=typ),
                "B_c": arr([0, 0, 1, 0], dtype=typ),
                "B_nan": arr([0, 0, 0, 1], dtype=typ),
            }
        ).sort_index(axis=1)

        # 使用测试框架验证实际结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)

        # 在不处理缺失值的情况下再次执行独热编码，验证结果是否只包含非缺失值列
        result = get_dummies(df, dummy_na=False, sparse=sparse, dtype=dtype)
        expected = expected[["C", "A_a", "A_b", "B_b", "B_c"]]
        tm.assert_frame_equal(result, expected)
    def test_dataframe_dummies_with_categorical(self, df, sparse, dtype):
        # 创建一个新的列 "cat"，其中包含分类数据 ["x", "y", "y"]
        df["cat"] = Categorical(["x", "y", "y"])
        
        # 调用 get_dummies 函数处理 DataFrame df，生成哑变量结果并按列名排序
        result = get_dummies(df, sparse=sparse, dtype=dtype).sort_index(axis=1)
        
        # 如果 sparse 参数为 True
        if sparse:
            # 设置数组类型为 SparseArray
            arr = SparseArray
            # 根据 dtype 的种类设置 SparseDtype 类型
            if dtype.kind == "b":
                typ = SparseDtype(dtype, False)
            else:
                typ = SparseDtype(dtype, 0)
        else:
            # 设置数组类型为 np.array
            arr = np.array
            # 类型为传入的 dtype
            typ = dtype

        # 创建一个预期的 DataFrame，包含预定义的列和数据
        expected = DataFrame(
            {
                "C": [1, 2, 3],
                "A_a": arr([1, 0, 1], dtype=typ),
                "A_b": arr([0, 1, 0], dtype=typ),
                "B_b": arr([1, 1, 0], dtype=typ),
                "B_c": arr([0, 0, 1], dtype=typ),
                "cat_x": arr([1, 0, 0], dtype=typ),
                "cat_y": arr([0, 1, 1], dtype=typ),
            }
        ).sort_index(axis=1)

        # 使用 assert_frame_equal 检查计算结果 result 和预期结果 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "get_dummies_kwargs,expected",
        [
            (
                {"data": DataFrame({"ä": ["a"]})},
                "ä_a",
            ),
            (
                {"data": DataFrame({"x": ["ä"]})},
                "x_ä",
            ),
            (
                {"data": DataFrame({"x": ["a"]}), "prefix": "ä"},
                "ä_a",
            ),
            (
                {"data": DataFrame({"x": ["a"]}), "prefix_sep": "ä"},
                "xäa",
            ),
        ],
    )
    def test_dataframe_dummies_unicode(self, get_dummies_kwargs, expected):
        # GH22084 get_dummies 函数在处理 DataFrame 列名中的 Unicode 字符时出现编码错误
        # 运行 get_dummies 函数并生成预期结果 DataFrame
        result = get_dummies(**get_dummies_kwargs)
        expected = DataFrame({expected: [True]})
        # 使用 assert_frame_equal 检查计算结果 result 和预期结果 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_get_dummies_basic_drop_first(self, sparse):
        # GH12402 添加一个新的参数 `drop_first` 以避免共线性问题
        # 基本情况
        # 创建字符列表并转换为 Series 对象
        s_list = list("abc")
        s_series = Series(s_list)
        # 创建指定索引的 Series 对象
        s_series_index = Series(s_list, list("ABC"))

        # 创建预期的 DataFrame，包含预定义的列和数据，指定数据类型为布尔型
        expected = DataFrame({"b": [0, 1, 0], "c": [0, 0, 1]}, dtype=bool)

        # 运行 get_dummies 函数，生成结果 DataFrame 并检查是否与预期结果相等
        result = get_dummies(s_list, drop_first=True, sparse=sparse)
        if sparse:
            expected = expected.apply(SparseArray, fill_value=False)
        tm.assert_frame_equal(result, expected)

        # 运行 get_dummies 函数，生成结果 DataFrame 并检查是否与预期结果相等
        result = get_dummies(s_series, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)

        # 设置预期结果的索引为指定的列表，并运行 get_dummies 函数，检查结果是否与预期一致
        expected.index = list("ABC")
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)
        tm.assert_frame_equal(result, expected)
    def test_get_dummies_basic_drop_first_one_level(self, sparse):
        # 测试分类变量只有一级的情况
        s_list = list("aaa")  # 创建一个包含字符列表的列表
        s_series = Series(s_list)  # 创建一个 Pandas Series 对象
        s_series_index = Series(s_list, list("ABC"))  # 创建一个带有指定索引的 Pandas Series 对象

        expected = DataFrame(index=RangeIndex(3))  # 创建一个预期的 DataFrame 对象，索引为 RangeIndex(3)

        result = get_dummies(s_list, drop_first=True, sparse=sparse)  # 调用 get_dummies 函数生成哑变量，设置 drop_first 和 sparse 参数
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果和预期的 DataFrame

        result = get_dummies(s_series, drop_first=True, sparse=sparse)  # 同上，对 Pandas Series 应用 get_dummies 函数
        tm.assert_frame_equal(result, expected)  # 同样使用测试工具比较结果和预期的 DataFrame

        expected = DataFrame(index=list("ABC"))  # 创建一个预期的 DataFrame 对象，索引为指定的列表
        result = get_dummies(s_series_index, drop_first=True, sparse=sparse)  # 同上，对带有指定索引的 Pandas Series 应用 get_dummies 函数
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果和预期的 DataFrame

    def test_get_dummies_basic_drop_first_NA(self, sparse):
        # 测试 NA 值处理和 drop_first 参数一起使用的情况
        s_NA = ["a", "b", np.nan]  # 创建一个包含 NA 值的列表
        res = get_dummies(s_NA, drop_first=True, sparse=sparse)  # 调用 get_dummies 函数生成哑变量，设置 drop_first 和 sparse 参数
        exp = DataFrame({"b": [0, 1, 0]}, dtype=bool)  # 创建一个预期的 DataFrame 对象
        if sparse:
            exp = exp.apply(SparseArray, fill_value=False)  # 如果 sparse 参数为 True，则将 DataFrame 转换为稀疏格式

        tm.assert_frame_equal(res, exp)  # 使用测试工具比较结果和预期的 DataFrame

        res_na = get_dummies(s_NA, dummy_na=True, drop_first=True, sparse=sparse)  # 同上，但同时设置 dummy_na 参数为 True
        exp_na = DataFrame({"b": [0, 1, 0], np.nan: [0, 0, 1]}, dtype=bool).reindex(
            ["b", np.nan], axis=1
        )  # 创建一个包含 NA 列的预期 DataFrame 对象
        if sparse:
            exp_na = exp_na.apply(SparseArray, fill_value=False)  # 如果 sparse 参数为 True，则将 DataFrame 转换为稀疏格式
        tm.assert_frame_equal(res_na, exp_na)  # 使用测试工具比较结果和预期的 DataFrame

        res_just_na = get_dummies(
            [np.nan], dummy_na=True, drop_first=True, sparse=sparse
        )  # 对仅包含一个 NA 值的列表应用 get_dummies 函数，同时设置 dummy_na 和 drop_first 参数为 True
        exp_just_na = DataFrame(index=RangeIndex(1))  # 创建一个预期的 DataFrame 对象，索引为 RangeIndex(1)
        tm.assert_frame_equal(res_just_na, exp_just_na)  # 使用测试工具比较结果和预期的 DataFrame

    def test_dataframe_dummies_drop_first(self, df, sparse):
        df = df[["A", "B"]]  # 从 DataFrame 中选择列 'A' 和 'B'，并重新赋值给 df
        result = get_dummies(df, drop_first=True, sparse=sparse)  # 调用 get_dummies 函数生成哑变量，设置 drop_first 和 sparse 参数
        expected = DataFrame({"A_b": [0, 1, 0], "B_c": [0, 0, 1]}, dtype=bool)  # 创建一个预期的 DataFrame 对象
        if sparse:
            expected = expected.apply(SparseArray, fill_value=False)  # 如果 sparse 参数为 True，则将 DataFrame 转换为稀疏格式
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果和预期的 DataFrame

    def test_dataframe_dummies_drop_first_with_categorical(self, df, sparse, dtype):
        df["cat"] = Categorical(["x", "y", "y"])  # 将分类变量 'cat' 添加到 DataFrame df 中
        result = get_dummies(df, drop_first=True, sparse=sparse)  # 调用 get_dummies 函数生成哑变量，设置 drop_first 和 sparse 参数
        expected = DataFrame(
            {"C": [1, 2, 3], "A_b": [0, 1, 0], "B_c": [0, 0, 1], "cat_y": [0, 1, 1]}
        )  # 创建一个预期的 DataFrame 对象
        cols = ["A_b", "B_c", "cat_y"]
        expected[cols] = expected[cols].astype(bool)  # 将指定列转换为布尔类型
        expected = expected[["C", "A_b", "B_c", "cat_y"]]  # 重新排序 DataFrame 的列
        if sparse:
            for col in cols:
                expected[col] = SparseArray(expected[col])  # 如果 sparse 参数为 True，则将 DataFrame 列转换为稀疏格式
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较结果和预期的 DataFrame
    # 测试函数：测试处理包含 NaN 值的 DataFrame 的 get_dummies 函数
    def test_dataframe_dummies_drop_first_with_na(self, df, sparse):
        # 将第三行所有列设置为 NaN
        df.loc[3, :] = [np.nan, np.nan, np.nan]
        # 使用 get_dummies 函数处理 DataFrame，生成虚拟变量并且删除第一个类别，根据 sparse 参数决定是否使用稀疏表示，然后按列名排序
        result = get_dummies(
            df, dummy_na=True, drop_first=True, sparse=sparse
        ).sort_index(axis=1)
        # 期望的 DataFrame 结果
        expected = DataFrame(
            {
                "C": [1, 2, 3, np.nan],
                "A_b": [0, 1, 0, 0],
                "A_nan": [0, 0, 0, 1],
                "B_c": [0, 0, 1, 0],
                "B_nan": [0, 0, 0, 1],
            }
        )
        # 将特定列转换为布尔值类型
        cols = ["A_b", "A_nan", "B_c", "B_nan"]
        expected[cols] = expected[cols].astype(bool)
        # 按列名排序结果 DataFrame
        expected = expected.sort_index(axis=1)
        # 如果 sparse 为 True，则将特定列转换为 SparseArray 类型
        if sparse:
            for col in cols:
                expected[col] = SparseArray(expected[col])

        # 使用 assert_frame_equal 函数比较计算结果和期望结果是否一致
        tm.assert_frame_equal(result, expected)

        # 使用 get_dummies 函数处理 DataFrame，生成虚拟变量，不处理 NaN 值，删除第一个类别，根据 sparse 参数决定是否使用稀疏表示
        result = get_dummies(df, dummy_na=False, drop_first=True, sparse=sparse)
        # 期望的 DataFrame 结果，仅包含特定列
        expected = expected[["C", "A_b", "B_c"]]
        # 使用 assert_frame_equal 函数比较计算结果和期望结果是否一致
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试处理整数和整数 Series 的 get_dummies 函数
    def test_get_dummies_int_int(self):
        # 创建整数 Series
        data = Series([1, 2, 1])
        # 使用 get_dummies 函数处理整数 Series，生成虚拟变量
        result = get_dummies(data)
        # 期望的 DataFrame 结果
        expected = DataFrame([[1, 0], [0, 1], [1, 0]], columns=[1, 2], dtype=bool)
        # 使用 assert_frame_equal 函数比较计算结果和期望结果是否一致
        tm.assert_frame_equal(result, expected)

        # 创建分类 Series
        data = Series(Categorical(["a", "b", "a"]))
        # 使用 get_dummies 函数处理分类 Series，生成虚拟变量
        result = get_dummies(data)
        # 期望的 DataFrame 结果
        expected = DataFrame(
            [[1, 0], [0, 1], [1, 0]], columns=Categorical(["a", "b"]), dtype=bool
        )
        # 使用 assert_frame_equal 函数比较计算结果和期望结果是否一致
        tm.assert_frame_equal(result, expected)

    # 测试函数：测试处理整数和分类 DataFrame 的 get_dummies 函数
    def test_get_dummies_int_df(self, dtype):
        # 创建整数和分类混合的 DataFrame
        data = DataFrame(
            {
                "A": [1, 2, 1],
                "B": Categorical(["a", "b", "a"]),
                "C": [1, 2, 1],
                "D": [1.0, 2.0, 1.0],
            }
        )
        # 期望的 DataFrame 列名
        columns = ["C", "D", "A_1", "A_2", "B_a", "B_b"]
        # 期望的 DataFrame 结果
        expected = DataFrame(
            [[1, 1.0, 1, 0, 1, 0], [2, 2.0, 0, 1, 0, 1], [1, 1.0, 1, 0, 1, 0]],
            columns=columns,
        )
        # 将特定列转换为指定数据类型
        expected[columns[2:]] = expected[columns[2:]].astype(dtype)
        # 使用 get_dummies 函数处理整数和分类 DataFrame，生成虚拟变量
        result = get_dummies(data, columns=["A", "B"], dtype=dtype)
        # 使用 assert_frame_equal 函数比较计算结果和期望结果是否一致
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("ordered", [True, False])
    # 测试函数：测试保留分类数据类型的 get_dummies 函数
    def test_dataframe_dummies_preserve_categorical_dtype(self, dtype, ordered):
        # 创建分类 Series
        cat = Categorical(list("xy"), categories=list("xyz"), ordered=ordered)
        # 使用 get_dummies 函数处理分类 Series，生成虚拟变量
        result = get_dummies(cat, dtype=dtype)

        # 期望的 DataFrame 数据
        data = np.array([[1, 0, 0], [0, 1, 0]], dtype=self.effective_dtype(dtype))
        # 期望的 DataFrame 列名
        cols = CategoricalIndex(
            cat.categories, categories=cat.categories, ordered=ordered
        )
        # 期望的 DataFrame 结果
        expected = DataFrame(data, columns=cols, dtype=self.effective_dtype(dtype))

        # 使用 assert_frame_equal 函数比较计算结果和期望结果是否一致
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("sparse", [True, False])
    # 测试用例：测试 get_dummies 函数不稀疏化所有列
    def test_get_dummies_dont_sparsify_all_columns(self, sparse):
        # GH18914
        # 创建一个包含 "GDP" 和 "Nation" 列的 DataFrame 对象
        df = DataFrame.from_dict({"GDP": [1, 2], "Nation": ["AB", "CD"]})
        # 对 "Nation" 列进行独热编码，根据 sparse 参数决定是否稀疏化
        df = get_dummies(df, columns=["Nation"], sparse=sparse)
        # 重新索引 DataFrame，只保留 "GDP" 列
        df2 = df.reindex(columns=["GDP"])
        # 使用测试工具比较两个 DataFrame 是否相等
        tm.assert_frame_equal(df[["GDP"]], df2)

    # 测试用例：测试 get_dummies 函数处理重复列名
    def test_get_dummies_duplicate_columns(self, df):
        # GH20839
        # 将 DataFrame 的列名全部设置为 "A"
        df.columns = ["A", "A", "A"]
        # 对 DataFrame 进行独热编码并按列名排序
        result = get_dummies(df).sort_index(axis=1)

        # 期望的 DataFrame，包含不同的独热编码列
        expected = DataFrame(
            [
                [1, True, False, True, False],
                [2, False, True, True, False],
                [3, True, False, False, True],
            ],
            columns=["A", "A_a", "A_b", "A_b", "A_c"],  # 列名包含了不同的编码后缀
        ).sort_index(axis=1)

        # 将 "A" 列的数据类型转换为 np.int64
        expected = expected.astype({"A": np.int64})

        # 使用测试工具比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试用例：测试 get_dummies 函数全部使用稀疏格式
    def test_get_dummies_all_sparse(self):
        # 创建一个包含 "A" 列的 DataFrame 对象
        df = DataFrame({"A": [1, 2]})
        # 对 "A" 列进行独热编码，指定使用稀疏格式
        result = get_dummies(df, columns=["A"], sparse=True)
        # 定义稀疏数组的数据类型
        dtype = SparseDtype("bool", False)
        # 期望的 DataFrame 包含两列稀疏数组
        expected = DataFrame(
            {
                "A_1": SparseArray([1, 0], dtype=dtype),
                "A_2": SparseArray([0, 1], dtype=dtype),
            }
        )
        # 使用测试工具比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试用例：测试 get_dummies 函数处理字符串值
    @pytest.mark.parametrize("values", ["baz"])
    def test_get_dummies_with_string_values(self, values):
        # issue #28383
        # 创建一个包含多列的 DataFrame 对象
        df = DataFrame(
            {
                "bar": [1, 2, 3, 4, 5, 6],
                "foo": ["one", "one", "one", "two", "two", "two"],
                "baz": ["A", "B", "C", "A", "B", "C"],
                "zoo": ["x", "y", "z", "q", "w", "t"],
            }
        )

        # 准备一个错误信息的字符串
        msg = "Input must be a list-like for parameter `columns`"

        # 使用 pytest 的断言检查是否抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            # 调用 get_dummies 函数，传入非列表形式的值作为 columns 参数
            get_dummies(df, columns=values)

    # 测试用例：测试 get_dummies 函数处理 Series 类型的数据
    def test_get_dummies_ea_dtype_series(self, any_numeric_ea_and_arrow_dtype):
        # GH#32430
        # 创建一个 Series 包含字符列表
        ser = Series(list("abca"))
        # 对 Series 进行独热编码，指定数据类型
        result = get_dummies(ser, dtype=any_numeric_ea_and_arrow_dtype)
        # 期望的 DataFrame 包含三列不同编码
        expected = DataFrame(
            {"a": [1, 0, 0, 1], "b": [0, 1, 0, 0], "c": [0, 0, 1, 0]},
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        # 使用测试工具比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试用例：测试 get_dummies 函数处理 DataFrame 类型的数据
    def test_get_dummies_ea_dtype_dataframe(self, any_numeric_ea_and_arrow_dtype):
        # GH#32430
        # 创建一个 DataFrame 包含单列 "x"，列值为字符列表
        df = DataFrame({"x": list("abca")})
        # 对 DataFrame 进行独热编码，指定数据类型
        result = get_dummies(df, dtype=any_numeric_ea_and_arrow_dtype)
        # 期望的 DataFrame 包含三列不同编码
        expected = DataFrame(
            {"x_a": [1, 0, 0, 1], "x_b": [0, 1, 0, 0], "x_c": [0, 0, 1, 0]},
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        # 使用测试工具比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    # 跳过测试：如果没有安装 pyarrow 模块
    @td.skip_if_no("pyarrow")
    # 定义测试方法，用于测试 get_dummies 函数处理不同数据类型的情况
    def test_get_dummies_ea_dtype(self):
        # 标识：GH#56273，测试用例的编号或标识
        # 遍历不同数据类型及其预期转换后的数据类型
        for dtype, exp_dtype in [
            ("string[pyarrow]", "boolean"),  # 字符串类型（使用 pyarrow）转换为布尔型
            ("string[pyarrow_numpy]", "bool"),  # 字符串类型（使用 pyarrow 和 numpy）转换为布尔型
            (CategoricalDtype(Index(["a"], dtype="string[pyarrow]")), "boolean"),  # 类别型数据转换为布尔型
            (CategoricalDtype(Index(["a"], dtype="string[pyarrow_numpy]")), "bool"),  # 类别型数据转换为布尔型
        ]:
            # 创建包含特定数据类型的 DataFrame，其中包含一列数据和一列固定的整数值
            df = DataFrame({"name": Series(["a"], dtype=dtype), "x": 1})
            # 对 DataFrame 应用 get_dummies 函数进行处理
            result = get_dummies(df)
            # 创建预期的 DataFrame，包含处理后的列数据及其预期的数据类型
            expected = DataFrame({"x": 1, "name_a": Series([True], dtype=exp_dtype)})
            # 使用测试框架检查 result 和 expected 是否相等
            tm.assert_frame_equal(result, expected)

    # 如果系统中没有安装 pyarrow，则跳过该测试用例
    @td.skip_if_no("pyarrow")
    def test_get_dummies_arrow_dtype(self):
        # 标识：GH#56273，测试用例的编号或标识
        # 创建包含特定数据类型的 DataFrame，其中包含一列数据和一列固定的整数值
        df = DataFrame({"name": Series(["a"], dtype=ArrowDtype(pa.string())), "x": 1})
        # 对 DataFrame 应用 get_dummies 函数进行处理
        result = get_dummies(df)
        # 创建预期的 DataFrame，包含处理后的列数据及其预期的数据类型
        expected = DataFrame({"x": 1, "name_a": Series([True], dtype="bool[pyarrow]")})
        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建包含类别数据类型的 DataFrame，其中包含一列数据和一列固定的整数值
        df = DataFrame(
            {
                "name": Series(
                    ["a"],
                    dtype=CategoricalDtype(Index(["a"], dtype=ArrowDtype(pa.string()))),
                ),
                "x": 1,
            }
        )
        # 对 DataFrame 应用 get_dummies 函数进行处理
        result = get_dummies(df)
        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```