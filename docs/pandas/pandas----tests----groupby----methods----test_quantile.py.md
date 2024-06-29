# `D:\src\scipysrc\pandas\pandas\tests\groupby\methods\test_quantile.py`

```
# 导入必要的库和模块
import numpy as np  # 导入 NumPy 库，用于处理数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，并指定别名 pd
from pandas import (  # 从 Pandas 库中导入 DataFrame 和 Index 类
    DataFrame,
    Index,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


@pytest.mark.parametrize(  # 使用 pytest.mark.parametrize 装饰器，参数化 interpolation 参数
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
@pytest.mark.parametrize(  # 参数化 a_vals 和 b_vals 参数
    "a_vals,b_vals",
    [
        # 不同类型的测试数据集：整数、浮点数、缺失数据、时间戳等
        ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]),  # 整数列表
        ([1, 2, 3, 4], [4, 3, 2, 1]),  # 整数列表
        ([1, 2, 3, 4, 5], [4, 3, 2, 1]),  # 整数列表
        ([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]),  # 浮点数列表
        ([1.0, np.nan, 3.0, np.nan, 5.0], [5.0, np.nan, 3.0, np.nan, 1.0]),  # 浮点数列表，包含缺失值
        ([np.nan, 4.0, np.nan, 2.0, np.nan], [np.nan, 4.0, np.nan, 2.0, np.nan]),  # 浮点数列表，全部为缺失值
        (
            pd.date_range("1/1/18", freq="D", periods=5),  # 时间戳序列
            pd.date_range("1/1/18", freq="D", periods=5)[::-1],  # 时间戳序列，反序
        ),
        (
            pd.date_range("1/1/18", freq="D", periods=5).as_unit("s"),  # 时间戳序列，单位为秒
            pd.date_range("1/1/18", freq="D", periods=5)[::-1].as_unit("s"),  # 时间戳序列，反序，单位为秒
        ),
        ([np.nan] * 5, [np.nan] * 5),  # 全部为缺失值的列表
    ],
)
@pytest.mark.parametrize("q", [0, 0.25, 0.5, 0.75, 1])  # 参数化 q 参数，用于指定分位数
def test_quantile(interpolation, a_vals, b_vals, q):
    # 合并 a_vals 和 b_vals 到一个 Series
    all_vals = pd.concat([pd.Series(a_vals), pd.Series(b_vals)])

    # 计算 Series a_vals 的分位数，并指定插值方法
    a_expected = pd.Series(a_vals).quantile(q, interpolation=interpolation)
    # 计算 Series b_vals 的分位数，并指定插值方法
    b_expected = pd.Series(b_vals).quantile(q, interpolation=interpolation)

    # 创建 DataFrame，包含 key 列和 all_vals 列
    df = DataFrame({"key": ["a"] * len(a_vals) + ["b"] * len(b_vals), "val": all_vals})

    # 创建预期的 DataFrame，包含期望的分位数值
    expected = DataFrame(
        [a_expected, b_expected], columns=["val"], index=Index(["a", "b"], name="key")
    )

    # 如果 all_vals 和 expected 的数据类型都是时间戳
    if all_vals.dtype.kind == "M" and expected.dtypes.values[0].kind == "M":
        # 这应该是不必要的，一旦 array_to_datetime 正确地从 Timestamp.unit 推断出非纳秒
        expected = expected.astype(all_vals.dtype)

    # 使用 groupby 操作计算分组后的分位数
    result = df.groupby("key").quantile(q, interpolation=interpolation)

    # 使用测试工具模块 tm，比较计算结果和预期结果是否相等
    tm.assert_frame_equal(result, expected)


def test_quantile_array():
    # 测试函数：https://github.com/pandas-dev/pandas/issues/27526
    df = DataFrame({"A": [0, 1, 2, 3, 4]})
    key = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    result = df.groupby(key).quantile([0.25])

    index = pd.MultiIndex.from_product([[0, 1], [0.25]])
    expected = DataFrame({"A": [0.25, 2.50]}, index=index)
    tm.assert_frame_equal(result, expected)

    df = DataFrame({"A": [0, 1, 2, 3], "B": [4, 5, 6, 7]})
    index = pd.MultiIndex.from_product([[0, 1], [0.25, 0.75]])

    key = np.array([0, 0, 1, 1], dtype=np.int64)
    result = df.groupby(key).quantile([0.25, 0.75])
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.25, 2.75], "B": [4.25, 4.75, 6.25, 6.75]}, index=index
    )
    tm.assert_frame_equal(result, expected)


def test_quantile_array2():
    # 测试函数：https://github.com/pandas-dev/pandas/pull/28085#issuecomment-524066959
    arr = np.random.default_rng(2).integers(0, 5, size=(10, 3), dtype=np.int64)
    # 使用给定的二维数组(arr)创建 DataFrame，列名为 "A", "B", "C"
    df = DataFrame(arr, columns=list("ABC"))
    # 按照列 "A" 进行分组，并计算分位数为 0.3 和 0.7 的值
    result = df.groupby("A").quantile([0.3, 0.7])
    # 创建期望的结果 DataFrame，包含列 "B" 和 "C" 的预期值
    expected = DataFrame(
        {
            "B": [2.0, 2.0, 2.3, 2.7, 0.3, 0.7, 3.2, 4.0, 0.3, 0.7],
            "C": [1.0, 1.0, 1.9, 3.0999999999999996, 0.3, 0.7, 2.6, 3.0, 1.2, 2.8],
        },
        # 使用 MultiIndex 创建索引，包含两个层级："A" 和空的第二层级
        index=pd.MultiIndex.from_product(
            [[0, 1, 2, 3, 4], [0.3, 0.7]], names=["A", None]
        ),
    )
    # 使用测试框架中的函数验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试未排序的数组的分位数计算
def test_quantile_array_no_sort():
    # 创建一个包含两列（A和B）的DataFrame
    df = DataFrame({"A": [0, 1, 2], "B": [3, 4, 5]})
    # 创建一个整数类型的NumPy数组作为分组键
    key = np.array([1, 0, 1], dtype=np.int64)
    # 使用不排序的方式，按照分位数（0.25, 0.5, 0.75）对DataFrame进行分组并计算分位数
    result = df.groupby(key, sort=False).quantile([0.25, 0.5, 0.75])
    # 创建预期的DataFrame，包含分组后的分位数计算结果
    expected = DataFrame(
        {"A": [0.5, 1.0, 1.5, 1.0, 1.0, 1.0], "B": [3.5, 4.0, 4.5, 4.0, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.25, 0.5, 0.75]]),
    )
    # 使用测试框架中的函数验证实际结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)

    # 再次按相同的不排序的方式，但计算不同顺序的分位数（0.75, 0.25）
    result = df.groupby(key, sort=False).quantile([0.75, 0.25])
    # 创建预期的DataFrame，包含按不同顺序计算的分位数结果
    expected = DataFrame(
        {"A": [1.5, 0.5, 1.0, 1.0], "B": [4.5, 3.5, 4.0, 4.0]},
        index=pd.MultiIndex.from_product([[1, 0], [0.75, 0.25]]),
    )
    # 使用测试框架中的函数验证实际结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于测试多级分组情况下的分位数计算
def test_quantile_array_multiple_levels():
    # 创建一个包含四列（A, B, c, d）的DataFrame
    df = DataFrame(
        {"A": [0, 1, 2], "B": [3, 4, 5], "c": ["a", "a", "a"], "d": ["a", "a", "b"]}
    )
    # 按多级索引（["c", "d"]）分组并计算分位数（0.25, 0.75）
    result = df.groupby(["c", "d"]).quantile([0.25, 0.75])
    # 创建预期的DataFrame，包含分组后的多级分位数计算结果
    index = pd.MultiIndex.from_tuples(
        [("a", "a", 0.25), ("a", "a", 0.75), ("a", "b", 0.25), ("a", "b", 0.75)],
        names=["c", "d", None],
    )
    expected = DataFrame(
        {"A": [0.25, 0.75, 2.0, 2.0], "B": [3.25, 3.75, 5.0, 5.0]}, index=index
    )
    # 使用测试框架中的函数验证实际结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)


# 使用参数化测试对不同的DataFrame大小、不同的分组方式和不同的分位数进行测试
@pytest.mark.parametrize("frame_size", [(2, 3), (100, 10)])
@pytest.mark.parametrize("groupby", [[0], [0, 1]])
@pytest.mark.parametrize("q", [[0.5, 0.6]])
def test_groupby_quantile_with_arraylike_q_and_int_columns(frame_size, groupby, q):
    # GH30289
    # 获取DataFrame的行数和列数
    nrow, ncol = frame_size
    # 创建一个DataFrame，包含整数列，并根据行数和列数进行初始化
    df = DataFrame(np.array([ncol * [_ % 4] for _ in range(nrow)]), columns=range(ncol))

    # 创建预期的多级索引
    idx_levels = [np.arange(min(nrow, 4))] * len(groupby) + [q]
    idx_codes = [[x for x in range(min(nrow, 4)) for _ in q]] * len(groupby) + [
        list(range(len(q))) * min(nrow, 4)
    ]
    expected_index = pd.MultiIndex(
        levels=idx_levels, codes=idx_codes, names=groupby + [None]
    )

    # 根据分组键进行分组并计算给定的分位数
    expected_values = [
        [float(x)] * (ncol - len(groupby)) for x in range(min(nrow, 4)) for _ in q
    ]
    expected_columns = [x for x in range(ncol) if x not in groupby]
    expected = DataFrame(
        expected_values, index=expected_index, columns=expected_columns
    )
    result = df.groupby(groupby).quantile(q)

    # 使用测试框架中的函数验证实际结果与预期结果是否一致
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，验证在DataFrame包含对象类型列时，计算分位数会引发TypeError异常
def test_quantile_raises():
    df = DataFrame([["foo", "a"], ["foo", "b"], ["foo", "c"]], columns=["key", "val"])

    # 使用pytest的断言验证计算分位数时会抛出TypeError异常
    with pytest.raises(TypeError, match="cannot be performed against 'object' dtypes"):
        df.groupby("key").quantile()


# 定义一个测试函数，验证在指定的分位数值超出范围时会引发ValueError异常
def test_quantile_out_of_bounds_q_raises():
    # https://github.com/pandas-dev/pandas/issues/27470
    df = DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": range(6)})
    g = df.groupby([0, 0, 0, 1, 1, 1])

    # 使用pytest的断言验证计算分位数时会抛出ValueError异常（分位数值为50和-1）
    with pytest.raises(ValueError, match="Got '50.0' instead"):
        g.quantile(50)

    with pytest.raises(ValueError, match="Got '-1.0' instead"):
        g.quantile(-1)


# 定义一个测试函数，验证在存在缺失的分组值时不会导致分段错误
def test_quantile_missing_group_values_no_segfaults():
    # GH 28662
    # 创建一个包含 NaN 值的 NumPy 数组
    data = np.array([1.0, np.nan, 1.0])
    # 使用 DataFrame 构造函数创建一个包含两列的数据框，一列是 "key"，另一列是从 0 到 2 的整数
    df = DataFrame({"key": data, "val": range(3)})

    # 按 "key" 列对数据框进行分组
    grp = df.groupby("key")
    # 循环执行 100 次，每次对分组后的数据框计算分位数，但是存在随机段错误（segfaults）的风险
    for _ in range(100):
        grp.quantile()
@pytest.mark.parametrize(
    "key, val, expected_key, expected_val",
    [
        # 第一个参数化测试用例
        ([1.0, np.nan, 3.0, np.nan], range(4), [1.0, 3.0], [0.0, 2.0]),
        # 第二个参数化测试用例
        ([1.0, np.nan, 2.0, 2.0], range(4), [1.0, 2.0], [0.0, 2.5]),
        # 第三个参数化测试用例
        (["a", "b", "b", np.nan], range(4), ["a", "b"], [0, 1.5]),
        # 第四个参数化测试用例
        ([0], [42], [0], [42.0]),
        # 第五个参数化测试用例
        ([], [], np.array([], dtype="float64"), np.array([], dtype="float64")),
    ],
)
def test_quantile_missing_group_values_correct_results(
    key, val, expected_key, expected_val
):
    # GH 28662, GH 33200, GH 33569
    # 创建一个 DataFrame 对象，包含 'key' 和 'val' 列
    df = DataFrame({"key": key, "val": val})

    # 创建期望的 DataFrame 对象，根据 'expected_val' 和 'expected_key' 初始化
    expected = DataFrame(
        expected_val, index=Index(expected_key, name="key"), columns=["val"]
    )

    # 按 'key' 列对 DataFrame 进行分组
    grp = df.groupby("key")

    # 计算分组后每组的中位数，并进行结果比较
    result = grp.quantile(0.5)
    tm.assert_frame_equal(result, expected)

    # 计算分组后每组的四分位数，并进行结果比较
    result = grp.quantile()
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "values",
    [
        # 第一个参数化测试用例
        pd.array([1, 0, None] * 2, dtype="Int64"),
        # 第二个参数化测试用例
        pd.array([True, False, None] * 2, dtype="boolean"),
    ],
)
@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
def test_groupby_quantile_nullable_array(values, q):
    # https://github.com/pandas-dev/pandas/issues/33136
    # 创建一个 DataFrame 对象，包含 'a' 和 'b' 列
    df = DataFrame({"a": ["x"] * 3 + ["y"] * 3, "b": values})
    # 计算每个分组的指定分位数，并进行结果比较
    result = df.groupby("a")["b"].quantile(q)

    if isinstance(q, list):
        # 根据不同的分位数列表创建多重索引
        idx = pd.MultiIndex.from_product((["x", "y"], q), names=["a", None])
        true_quantiles = [0.0, 0.5, 1.0]
    else:
        # 创建索引对象
        idx = Index(["x", "y"], name="a")
        true_quantiles = [0.5]

    # 创建期望的 Series 对象，根据 true_quantiles 和 idx 初始化
    expected = pd.Series(true_quantiles * 2, index=idx, name="b", dtype="Float64")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("q", [0.5, [0.0, 0.5, 1.0]])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_quantile_raises_on_invalid_dtype(q, numeric_only):
    # 创建一个 DataFrame 对象，包含 'a', 'b', 'c' 列
    df = DataFrame({"a": [1], "b": [2.0], "c": ["x"]})
    if numeric_only:
        # 计算每个分组的指定分位数，并进行结果比较
        result = df.groupby("a").quantile(q, numeric_only=numeric_only)
        expected = df.groupby("a")[["b"]].quantile(q)
        tm.assert_frame_equal(result, expected)
    else:
        # 检查异常抛出情况
        with pytest.raises(
            TypeError, match="'quantile' cannot be performed against 'object' dtypes!"
        ):
            df.groupby("a").quantile(q, numeric_only=numeric_only)


def test_groupby_quantile_NA_float(any_float_dtype):
    # GH#42849
    # 创建一个 DataFrame 对象，包含 'x' 和 'y' 列
    df = DataFrame({"x": [1, 1], "y": [0.2, np.nan]}, dtype=any_float_dtype)
    # 计算每个分组的指定分位数，并进行结果比较
    result = df.groupby("x")["y"].quantile(0.5)
    # 创建索引对象，根据 any_float_dtype 初始化
    exp_index = Index([1.0], dtype=any_float_dtype, name="x")

    if any_float_dtype in ["Float32", "Float64"]:
        expected_dtype = any_float_dtype
    else:
        expected_dtype = None

    # 创建期望的 Series 对象，根据 expected_dtype 和 exp_index 初始化
    expected = pd.Series([0.2], dtype=expected_dtype, index=exp_index, name="y")
    tm.assert_series_equal(result, expected)

    # 计算每个分组的多个指定分位数
    result = df.groupby("x")["y"].quantile([0.5, 0.75])
    # 创建一个预期的 Pandas Series 对象，其中包含两个元素，值均为 0.2
    # Series 对象的索引使用了 MultiIndex，通过 from_product 方法生成，使用 exp_index 与 [0.5, 0.75] 的笛卡尔积，索引层级名为 "x" 和 None
    # Series 对象的名称设置为 "y"，数据类型为 expected_dtype
    expected = pd.Series(
        [0.2] * 2,
        index=pd.MultiIndex.from_product((exp_index, [0.5, 0.75]), names=["x", None]),
        name="y",
        dtype=expected_dtype,
    )
    
    # 使用测试工具集中的 assert_series_equal 方法，比较 result 和 expected 两个 Pandas Series 对象是否相等
    tm.assert_series_equal(result, expected)
# GH#42849
def test_groupby_quantile_NA_int(any_int_ea_dtype):
    # 创建一个包含两列的 DataFrame，列 'x' 包含值 [1, 1]，列 'y' 包含值 [2, 5]，数据类型由参数 any_int_ea_dtype 决定
    df = DataFrame({"x": [1, 1], "y": [2, 5]}, dtype=any_int_ea_dtype)
    # 对 DataFrame 按列 'x' 进行分组，计算 'y' 列的中位数（分位数）为 0.5 的值
    result = df.groupby("x")["y"].quantile(0.5)
    # 创建预期结果的 Series 对象，包含值 [3.5]，数据类型为 "Float64"，索引为以 any_int_ea_dtype 类型命名的 Index 对象
    expected = pd.Series(
        [3.5],
        dtype="Float64",
        index=Index([1], name="x", dtype=any_int_ea_dtype),
        name="y",
    )
    # 使用测试工具函数 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(expected, result)

    # 对 DataFrame 按列 'x' 进行分组，计算所有列的中位数（分位数）为 0.5 的值
    result = df.groupby("x").quantile(0.5)
    # 创建预期结果的 DataFrame 对象，包含值 {"y": 3.5}，数据类型为 "Float64"，索引为以 any_int_ea_dtype 类型命名的 Index 对象
    expected = DataFrame(
        {"y": 3.5}, dtype="Float64", index=Index([1], name="x", dtype=any_int_ea_dtype)
    )
    # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "interpolation, val1, val2", [("lower", 2, 2), ("higher", 2, 3), ("nearest", 2, 2)]
)
def test_groupby_quantile_all_na_group_masked(
    interpolation, val1, val2, any_numeric_ea_dtype
):
    # GH#37493
    # 创建一个包含两列的 DataFrame，列 'a' 包含值 [1, 1, 1, 2]，列 'b' 包含值 [1, 2, 3, pd.NA]，数据类型由参数 any_numeric_ea_dtype 决定
    df = DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 3, pd.NA]}, dtype=any_numeric_ea_dtype
    )
    # 对 DataFrame 按列 'a' 进行分组，计算 'b' 列的分位数为 [0.5, 0.7] 的值，使用指定的插值方法 interpolation
    result = df.groupby("a").quantile(q=[0.5, 0.7], interpolation=interpolation)
    # 创建预期结果的 DataFrame 对象，包含值 {"b": [val1, val2, pd.NA, pd.NA]}，数据类型为 any_numeric_ea_dtype，索引为 MultiIndex 对象
    # MultiIndex 包含两个级别，分别为列 'a' 的值 [1, 1, 2, 2] 和分位数 [0.5, 0.7]
    expected = DataFrame(
        {"b": [val1, val2, pd.NA, pd.NA]},
        dtype=any_numeric_ea_dtype,
        index=pd.MultiIndex.from_arrays(
            [pd.Series([1, 1, 2, 2], dtype=any_numeric_ea_dtype), [0.5, 0.7, 0.5, 0.7]],
            names=["a", None],
        ),
    )
    # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("interpolation", ["midpoint", "linear"])
def test_groupby_quantile_all_na_group_masked_interp(
    interpolation, any_numeric_ea_dtype
):
    # GH#37493
    # 创建一个包含两列的 DataFrame，列 'a' 包含值 [1, 1, 1, 2]，列 'b' 包含值 [1, 2, 3, pd.NA]，数据类型由参数 any_numeric_ea_dtype 决定
    df = DataFrame(
        {"a": [1, 1, 1, 2], "b": [1, 2, 3, pd.NA]}, dtype=any_numeric_ea_dtype
    )
    # 对 DataFrame 按列 'a' 进行分组，计算 'b' 列的分位数为 [0.5, 0.75] 的值，使用指定的插值方法 interpolation
    result = df.groupby("a").quantile(q=[0.5, 0.75], interpolation=interpolation)

    # 根据 any_numeric_ea_dtype 的值确定预期结果的数据类型
    if any_numeric_ea_dtype == "Float32":
        expected_dtype = any_numeric_ea_dtype
    else:
        expected_dtype = "Float64"

    # 创建预期结果的 DataFrame 对象，包含值 {"b": [2.0, 2.5, pd.NA, pd.NA]}，数据类型为 expected_dtype，索引为 MultiIndex 对象
    # MultiIndex 包含两个级别，分别为列 'a' 的值 [1, 1, 2, 2] 和分位数 [0.5, 0.75]
    expected = DataFrame(
        {"b": [2.0, 2.5, pd.NA, pd.NA]},
        dtype=expected_dtype,
        index=pd.MultiIndex.from_arrays(
            [
                pd.Series([1, 1, 2, 2], dtype=any_numeric_ea_dtype),
                [0.5, 0.75, 0.5, 0.75],
            ],
            names=["a", None],
        ),
    )
    # 使用测试工具函数 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("dtype", ["Float64", "Float32"])
def test_groupby_quantile_allNA_column(dtype):
    # GH#42849
    # 创建一个包含两列的 DataFrame，列 'x' 包含值 [1, 1]，列 'y' 包含值 [pd.NA, pd.NA]，数据类型由参数 dtype 决定
    df = DataFrame({"x": [1, 1], "y": [pd.NA] * 2}, dtype=dtype)
    # 对 DataFrame 按列 'x' 进行分组，计算 'y' 列的中位数（分位数）为 0.5 的值
    result = df.groupby("x")["y"].quantile(0.5)
    # 创建预期结果的 Series 对象，包含值 [np.nan]，数据类型为 dtype，索引为以 dtype 类型命名的 Index 对象
    expected = pd.Series(
        [np.nan], dtype=dtype, index=Index([1.0], dtype=dtype), name="y"
    )
    expected.index.name = "x"
    # 使用测试工具函数 tm.assert_series_equal 检查 result 和 expected 是否相等
    tm.assert_series_equal(expected, result)


def test_groupby_timedelta_quantile():
    # GH: 29485
    # 创建一个包含两列的 DataFrame，列 'value' 包含从 0 开始递增的 timedelta 值，单位为秒，列 'group' 包含值 [1, 1, 2, 2]
    df = DataFrame(
        {"value": pd.to_timedelta(np.arange(4), unit="s"), "group": [1, 1, 2, 2]}
    )
    # 对 DataFrame 按列 'group' 进行分组，计算所有列的分位数为 0.99 的值
    result = df.groupby("group").quantile(0.99)
    # 创建预期的DataFrame对象，包含一个"value"列，列值是两个Timedelta对象
    expected = DataFrame(
        {
            "value": [
                pd.Timedelta("0 days 00:00:00.990000"),  # 第一个Timedelta对象，表示0.99秒
                pd.Timedelta("0 days 00:00:02.990000"),  # 第二个Timedelta对象，表示2.99秒
            ]
        },
        index=Index([1, 2], name="group"),  # 指定索引为整数1和2，索引名称为"group"
    )
    # 使用pytest的assert_frame_equal方法比较result和expected两个DataFrame对象是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于按时间戳分组并计算分位数
def test_timestamp_groupby_quantile(unit):
    # GH 33168
    # 创建一个包含100个日期时间的日期范围，每分钟一个数据点，以UTC时区为基准，按指定的时间单位进行取整到小时
    dti = pd.date_range(
        start="2020-04-19 00:00:00", freq="1min", periods=100, tz="UTC", unit=unit
    ).floor("1h")
    # 创建一个DataFrame，包含'timestamp'列、'category'列和'value'列，分别为日期时间、序号和值
    df = DataFrame(
        {
            "timestamp": dti,
            "category": list(range(1, 101)),
            "value": list(range(101, 201)),
        }
    )

    # 对DataFrame按'timestamp'列进行分组，并计算分位数为0.2和0.8的值
    result = df.groupby("timestamp").quantile([0.2, 0.8])

    # 创建一个期望的DataFrame，包含对应的索引和预期的数值
    mi = pd.MultiIndex.from_product([dti[::99], [0.2, 0.8]], names=("timestamp", None))
    expected = DataFrame(
        [
            {"category": 12.8, "value": 112.8},
            {"category": 48.2, "value": 148.2},
            {"category": 68.8, "value": 168.8},
            {"category": 92.2, "value": 192.2},
        ],
        index=mi,
    )

    # 使用测试工具比较计算结果和预期结果
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于按照不同时间序列和时区进行分组并计算分位数
def test_groupby_quantile_dt64tz_period():
    # GH#51373
    # 创建一个包含1000个日期时间的日期范围
    dti = pd.date_range("2016-01-01", periods=1000)
    # 将日期时间转换为DataFrame，并复制为df
    df = pd.Series(dti).to_frame().copy()
    # 添加时区信息到第二列
    df[1] = dti.tz_localize("US/Pacific")
    # 将日期时间转换为周期
    df[2] = dti.to_period("D")
    # 计算日期时间与第一个日期时间的差值并加入到第三列
    df[3] = dti - dti[0]
    # 将最后一行置为NaT（Not a Time）
    df.iloc[-1] = pd.NaT

    # 使用np.tile对数组进行重复以构建分组标签by
    by = np.tile(np.arange(5), 200)
    # 对df按by进行分组
    gb = df.groupby(by)

    # 对分组后的数据计算分位数为0.5
    result = gb.quantile(0.5)

    # 检查分组计算的结果与预期结果是否匹配
    exp = {i: df.iloc[i::5].quantile(0.5) for i in range(5)}
    expected = DataFrame(exp).T.infer_objects()
    expected.index = expected.index.astype(int)

    # 使用测试工具比较计算结果和预期结果
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于按多级索引进行分组并计算分位数，同时测试索引级别的顺序
def test_groupby_quantile_nonmulti_levels_order():
    # Non-regression test for GH #53009
    # 创建一个多级索引
    ind = pd.MultiIndex.from_tuples(
        [
            (0, "a", "B"),
            (0, "a", "A"),
            (0, "b", "B"),
            (0, "b", "A"),
            (1, "a", "B"),
            (1, "a", "A"),
            (1, "b", "B"),
            (1, "b", "A"),
        ],
        names=["sample", "cat0", "cat1"],
    )
    # 创建一个Series，使用上述多级索引
    ser = pd.Series(range(8), index=ind)
    # 对Series按'cat1'级别进行分组，并计算分位数为0.2和0.8
    result = ser.groupby(level="cat1", sort=False).quantile([0.2, 0.8])

    # 创建一个期望的Series，包含对应的索引和预期的数值
    qind = pd.MultiIndex.from_tuples(
        [("B", 0.2), ("B", 0.8), ("A", 0.2), ("A", 0.8)], names=["cat1", None]
    )
    expected = pd.Series([1.2, 4.8, 2.2, 5.8], index=qind)

    # 使用测试工具比较计算结果和预期结果
    tm.assert_series_equal(result, expected)

    # 检查索引级别是否没有被排序
    expected_levels = pd.core.indexes.frozen.FrozenList([["B", "A"], [0.2, 0.8]])
    tm.assert_equal(result.index.levels, expected_levels)
```