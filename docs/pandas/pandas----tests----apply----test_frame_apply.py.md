# `D:\src\scipysrc\pandas\pandas\tests\apply\test_frame_apply.py`

```
# 从 datetime 模块导入 datetime 类
from datetime import datetime
# 导入警告模块
import warnings

# 导入 numpy 库，并使用 np 别名
import numpy as np
# 导入 pytest 测试框架
import pytest

# 从 pandas 库的 core.dtypes.dtypes 模块中导入 CategoricalDtype 类型
from pandas.core.dtypes.dtypes import CategoricalDtype

# 导入 pandas 库，并使用 pd 别名
import pandas as pd
# 从 pandas 库中导入 DataFrame, MultiIndex, Series, Timestamp, date_range 函数
from pandas import (
    DataFrame,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)
# 导入 pandas 内部测试工具模块
import pandas._testing as tm
# 从 pandas 测试模块中导入 zip_frames 函数
from pandas.tests.frame.common import zip_frames


@pytest.fixture
def int_frame_const_col():
    """
    Fixture for DataFrame of ints which are constant per column

    Columns are ['A', 'B', 'C'], with values (per column): [1, 2, 3]
    """
    # 创建一个 DataFrame，包含三列（'A', 'B', 'C'），每列的值为 [1, 2, 3]
    df = DataFrame(
        np.tile(np.arange(3, dtype="int64"), 6).reshape(6, -1) + 1,
        columns=["A", "B", "C"],
    )
    return df


@pytest.fixture(params=["python", pytest.param("numba", marks=pytest.mark.single_cpu)])
def engine(request):
    # 如果参数为 'numba'，则检查是否导入 numba 模块
    if request.param == "numba":
        pytest.importorskip("numba")
    # 返回参数值
    return request.param


def test_apply(float_frame, engine, request):
    # 如果引擎为 'numba'，则标记测试为预期失败，原因是 'numba' 引擎尚不支持 numpy 的 ufunc
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine not supporting numpy ufunc yet")
        request.node.add_marker(mark)
    # 忽略所有 numpy 错误
    with np.errstate(all="ignore"):
        # 对 Series 应用 numpy 的 sqrt 函数
        result = np.sqrt(float_frame["A"])
        # 期望的结果，使用 float_frame 的 apply 方法应用 numpy 的 sqrt 函数在引擎上
        expected = float_frame.apply(np.sqrt, engine=engine)["A"]
        # 断言 Series 相等
        tm.assert_series_equal(result, expected)

        # 对列应用 numpy 的均值函数
        result = float_frame.apply(np.mean, engine=engine)["A"]
        # 期望的结果，计算 float_frame['A'] 的均值
        expected = np.mean(float_frame["A"])
        assert result == expected

        # 获取 float_frame 的第一个索引，并计算沿轴的均值
        d = float_frame.index[0]
        result = float_frame.apply(np.mean, axis=1, engine=engine)
        # 期望的结果，计算 float_frame 在日期 d 上的均值
        expected = np.mean(float_frame.xs(d))
        assert result[d] == expected
        assert result.index is float_frame.index


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("nopython", [True, False])
def test_apply_args(float_frame, axis, raw, engine, nopython):
    # 设置引擎的关键字参数
    engine_kwargs = {"nopython": nopython}
    # 应用 lambda 函数来对 float_frame 应用加法操作
    result = float_frame.apply(
        lambda x, y: x + y,
        axis,
        args=(1,),
        raw=raw,
        engine=engine,
        engine_kwargs=engine_kwargs,
    )
    # 期望的结果，float_frame 中的所有元素加 1
    expected = float_frame + 1
    tm.assert_frame_equal(result, expected)

    # GH:58712
    # 应用 lambda 函数来对 float_frame 应用加法操作，带有两个额外的参数 a 和 b
    result = float_frame.apply(
        lambda x, a, b: x + a + b,
        args=(1,),
        b=2,
        raw=raw,
        engine=engine,
        engine_kwargs=engine_kwargs,
    )
    # 期望的结果，float_frame 中的所有元素加 3
    expected = float_frame + 3
    tm.assert_frame_equal(result, expected)
    # 如果使用的是 "numba" 引擎
    # 检查 lambda 函数中的关键字参数是否被支持，因为 numba 不支持关键字参数
    with pytest.raises(
        pd.errors.NumbaUtilError,
        match="numba does not support keyword-only arguments",
    ):
        # 对 float_frame 应用 lambda 函数，尝试使用带有关键字参数的形式
        float_frame.apply(
            lambda x, a, *, b: x + a + b,  # lambda 函数定义：x, a 是位置参数，b 是关键字参数
            args=(1,),                    # lambda 函数的位置参数
            b=2,                          # lambda 函数的关键字参数
            raw=raw,                      # 应用的原始数据标志
            engine=engine,                # 使用的计算引擎
            engine_kwargs=engine_kwargs,  # 引擎的其他参数
        )

    # 如果使用的是 "numba" 引擎
    # 再次检查 lambda 函数中的关键字参数是否被支持
    with pytest.raises(
        pd.errors.NumbaUtilError,
        match="numba does not support keyword-only arguments",
    ):
        # 对 float_frame 应用 lambda 函数，尝试使用带有关键字参数的形式
        float_frame.apply(
            lambda *x, b: x[0] + x[1] + b,  # lambda 函数定义：*x 表示任意数量的位置参数，b 是关键字参数
            args=(1,),                    # lambda 函数的位置参数
            b=2,                          # lambda 函数的关键字参数
            raw=raw,                      # 应用的原始数据标志
            engine=engine,                # 使用的计算引擎
            engine_kwargs=engine_kwargs,  # 引擎的其他参数
        )
def test_apply_categorical_func():
    # GH 9573
    # 创建一个包含两列的DataFrame，每列包含两个类别值
    df = DataFrame({"c0": ["A", "A", "B", "B"], "c1": ["C", "C", "D", "D"]})
    # 将DataFrame中的每列转换为分类类型，并返回结果
    result = df.apply(lambda ts: ts.astype("category"))

    # 断言结果DataFrame的形状为 (4, 2)
    assert result.shape == (4, 2)
    # 断言结果DataFrame的"c0"列的数据类型为CategoricalDtype
    assert isinstance(result["c0"].dtype, CategoricalDtype)
    # 断言结果DataFrame的"c1"列的数据类型为CategoricalDtype
    assert isinstance(result["c1"].dtype, CategoricalDtype)


def test_apply_axis1_with_ea():
    # GH#36785
    # 创建一个包含单个列"A"的DataFrame，列值为具有时区信息的时间戳
    expected = DataFrame({"A": [Timestamp("2013-01-01", tz="UTC")]})
    # 应用一个lambda函数在axis=1上，返回结果
    result = expected.apply(lambda x: x, axis=1)
    # 使用测试工具断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "data, dtype",
    [(1, None), (1, CategoricalDtype([1])), (Timestamp("2013-01-01", tz="UTC"), None)],
)
def test_agg_axis1_duplicate_index(data, dtype):
    # GH 42380
    # 创建一个包含重复索引"a"的DataFrame，每行包含单个值data，指定数据类型dtype
    expected = DataFrame([[data], [data]], index=["a", "a"], dtype=dtype)
    # 应用一个lambda函数在axis=1上，返回结果
    result = expected.agg(lambda x: x, axis=1)
    # 使用测试工具断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_apply_mixed_datetimelike():
    # mixed datetimelike
    # GH 7778
    # 创建一个包含两列"A"和"B"的DataFrame，分别包含日期范围和时间增量
    expected = DataFrame(
        {
            "A": date_range("20130101", periods=3),
            "B": pd.to_timedelta(np.arange(3), unit="s"),
        }
    )
    # 应用一个lambda函数在axis=1上，返回结果
    result = expected.apply(lambda x: x, axis=1)
    # 使用测试工具断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("func", [np.sqrt, np.mean])
def test_apply_empty(func, engine):
    # empty
    # 创建一个空的DataFrame
    empty_frame = DataFrame()

    # 应用给定的函数func到空的DataFrame，使用指定的引擎
    result = empty_frame.apply(func, engine=engine)
    # 断言结果DataFrame为空
    assert result.empty


def test_apply_float_frame(float_frame, engine):
    # 创建一个没有行的DataFrame，并应用lambda函数计算列的均值，使用指定的引擎
    no_rows = float_frame[:0]
    result = no_rows.apply(lambda x: x.mean(), engine=engine)
    # 创建一个没有列的DataFrame，并应用lambda函数计算行的均值，使用指定的引擎
    no_cols = float_frame.loc[:, []]
    result = no_cols.apply(lambda x: x.mean(), axis=1, engine=engine)
    # 断言结果Series包含NaN，并与预期结果相等
    expected = Series(np.nan, index=float_frame.index)
    tm.assert_series_equal(result, expected)


def test_apply_empty_except_index(engine):
    # GH 2476
    # 创建一个只有索引"a"的DataFrame
    expected = DataFrame(index=["a"])
    # 应用一个lambda函数获取每行中索引为"a"的值，使用指定的引擎
    result = expected.apply(lambda x: x["a"], axis=1, engine=engine)
    # 使用测试工具断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_apply_with_reduce_empty():
    # reduce with an empty DataFrame
    # 创建一个空的DataFrame
    empty_frame = DataFrame()

    x = []
    # 应用一个lambda函数x.append到空DataFrame的每行，以"expand"方式扩展结果
    result = empty_frame.apply(x.append, axis=1, result_type="expand")
    # 使用测试工具断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, empty_frame)
    # 应用一个lambda函数x.append到空DataFrame的每行，以"reduce"方式聚合结果
    result = empty_frame.apply(x.append, axis=1, result_type="reduce")
    # 创建一个空的Series作为预期结果
    expected = Series([], dtype=np.float64)
    # 使用测试工具断言结果Series与预期Series相等
    tm.assert_series_equal(result, expected)

    # 创建一个空DataFrame，列为["a", "b", "c"]
    empty_with_cols = DataFrame(columns=["a", "b", "c"])
    # 应用一个lambda函数x.append到空DataFrame的每行，以"expand"方式扩展结果
    result = empty_with_cols.apply(x.append, axis=1, result_type="expand")
    # 使用测试工具断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, empty_with_cols)
    # 应用一个lambda函数x.append到空DataFrame的每行，以"reduce"方式聚合结果
    result = empty_with_cols.apply(x.append, axis=1, result_type="reduce")
    # 创建一个空的Series作为预期结果
    expected = Series([], dtype=np.float64)
    # 使用测试工具断言结果Series与预期Series相等
    tm.assert_series_equal(result, expected)

    # 确保未调用x.append函数
    assert x == []
def test_apply_funcs_over_empty(func):
    # GH 28213
    # 创建一个空的 DataFrame，列名为 ["a", "b", "c"]
    df = DataFrame(columns=["a", "b", "c"])

    # 对空的 DataFrame 应用 np 模块中的函数 func，并保存结果
    result = df.apply(getattr(np, func))
    
    # 获取 DataFrame 自身方法 func 的期望结果
    expected = getattr(df, func)()
    
    # 如果 func 是 "sum" 或 "prod"，将期望结果转换为 float 类型
    if func in ("sum", "prod"):
        expected = expected.astype(float)
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_series_equal(result, expected)


def test_nunique_empty():
    # GH 28213
    # 创建一个空的 DataFrame，列名为 ["a", "b", "c"]
    df = DataFrame(columns=["a", "b", "c"])

    # 计算空 DataFrame 中每列的唯一值数量
    result = df.nunique()
    
    # 创建一个与 DataFrame 列名对应的 Series，值全部为 0 作为期望结果
    expected = Series(0, index=df.columns)
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_series_equal(result, expected)

    # 对 DataFrame 进行转置，计算转置后每行的唯一值数量
    result = df.T.nunique()
    
    # 创建一个空的 Series 作为期望结果，数据类型为 np.float64
    expected = Series([], dtype=np.float64)
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_series_equal(result, expected)


def test_apply_standard_nonunique():
    # 创建一个包含重复索引的 DataFrame
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["a", "a", "c"])

    # 对 DataFrame 应用 lambda 函数，按行取每行的第一个元素作为结果
    result = df.apply(lambda s: s[0], axis=1)
    
    # 创建一个与结果对应的 Series 作为期望结果
    expected = Series([1, 4, 7], ["a", "a", "c"])
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_series_equal(result, expected)

    # 对 DataFrame 进行转置，对转置后的 DataFrame 应用 lambda 函数，按列取每列的第一个元素作为结果
    result = df.T.apply(lambda s: s[0], axis=0)
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_series_equal(result, expected)


def test_apply_broadcast_scalars(float_frame):
    # scalars
    # 对 DataFrame 的每列应用 np.mean 函数，广播结果
    result = float_frame.apply(np.mean, result_type="broadcast")
    
    # 创建一个期望结果 DataFrame，包含每列的平均值，索引与 float_frame 相同
    expected = DataFrame([float_frame.mean()], index=float_frame.index)
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_scalars_axis1(float_frame):
    # 对 DataFrame 的每行应用 np.mean 函数，广播结果
    result = float_frame.apply(np.mean, axis=1, result_type="broadcast")
    
    # 计算每行的平均值
    m = float_frame.mean(axis=1)
    
    # 创建一个期望结果 DataFrame，每列包含相同的平均值
    expected = DataFrame({c: m for c in float_frame.columns})
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_lists_columns(float_frame):
    # lists
    # 对 DataFrame 的每行应用 lambda 函数，返回列名列表
    result = float_frame.apply(
        lambda x: list(range(len(float_frame.columns))),
        axis=1,
        result_type="broadcast",
    )
    
    # 创建一个期望结果 DataFrame，每行包含相同的列名列表
    m = list(range(len(float_frame.columns)))
    expected = DataFrame(
        [m] * len(float_frame.index),
        dtype="float64",
        index=float_frame.index,
        columns=float_frame.columns,
    )
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_lists_index(float_frame):
    # 对 DataFrame 的每列应用 lambda 函数，返回行索引列表
    result = float_frame.apply(
        lambda x: list(range(len(float_frame.index))), result_type="broadcast"
    )
    
    # 创建一个期望结果 DataFrame，每列包含相同的行索引列表
    m = list(range(len(float_frame.index)))
    expected = DataFrame(
        {c: m for c in float_frame.columns},
        dtype="float64",
        index=float_frame.index,
    )
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_frame_equal(result, expected)


def test_apply_broadcast_list_lambda_func(int_frame_const_col):
    # preserve columns
    # 对 DataFrame 应用 lambda 函数，每行返回固定列表 [1, 2, 3]
    df = int_frame_const_col
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type="broadcast")
    
    # 使用测试框架检查结果和 DataFrame 是否相等
    tm.assert_frame_equal(result, df)


def test_apply_broadcast_series_lambda_func(int_frame_const_col):
    # 对 DataFrame 应用 lambda 函数，每行返回固定 Series
    df = int_frame_const_col
    result = df.apply(
        lambda x: Series([1, 2, 3], index=list("abc")),
        axis=1,
        result_type="broadcast",
    )
    
    # 创建一个与 df 相同的 DataFrame 作为期望结果
    expected = df.copy()
    
    # 使用测试框架检查结果和期望结果是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_float_frame(float_frame, axis, engine):
    # 此测试用例参数化 axis，测试 DataFrame 的原始应用情况
    # 这里不添加具体注释，因为这是一个参数化测试，需要在测试框架中运行以验证不同的 axis 参数
    # 如果使用的引擎是 "numba"，跳过测试并显示相应消息，因为 numba 无法处理用户定义函数返回 None 的情况。
    if engine == "numba":
        pytest.skip("numba can't handle when UDF returns None.")

    # 定义一个内部函数 _assert_raw，用于断言参数 x 是一个 numpy 数组，并且是一维的。
    def _assert_raw(x):
        assert isinstance(x, np.ndarray)  # 断言 x 是 numpy 数组
        assert x.ndim == 1  # 断言 x 是一维数组

    # 对 float_frame 应用 _assert_raw 函数，沿指定轴进行操作，使用指定的引擎执行，同时以原始数据格式处理。
    float_frame.apply(_assert_raw, axis=axis, engine=engine, raw=True)
# 使用 pytest 的参数化装饰器指定 axis 参数为 0 和 1 进行测试
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_float_frame_lambda(float_frame, axis, engine):
    # 对 float_frame 应用 np.mean 函数，设置 raw=True，使用指定的引擎进行计算
    result = float_frame.apply(np.mean, axis=axis, engine=engine, raw=True)
    # 对 float_frame 应用 lambda 函数计算每列的均值，作为预期结果
    expected = float_frame.apply(lambda x: x.values.mean(), axis=axis)
    # 断言结果与预期结果是否相等
    tm.assert_series_equal(result, expected)


def test_apply_raw_float_frame_no_reduction(float_frame, engine):
    # no reduction
    # 对 float_frame 应用 lambda 函数将每个元素乘以 2，设置 raw=True，使用指定的引擎进行计算
    result = float_frame.apply(lambda x: x * 2, engine=engine, raw=True)
    # 预期结果为 float_frame 中的每个元素乘以 2
    expected = float_frame * 2
    # 断言结果与预期结果是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("axis", [0, 1])
def test_apply_raw_mixed_type_frame(axis, engine):
    if engine == "numba":
        pytest.skip("isinstance check doesn't work with numba")

    def _assert_raw(x):
        # 断言 x 是 np.ndarray 类型，并且是一维数组
        assert isinstance(x, np.ndarray)
        assert x.ndim == 1

    # 创建一个包含不同数据类型的 DataFrame
    df = DataFrame(
        {
            "a": 1.0,
            "b": 2,
            "c": "foo",
            "float32": np.array([1.0] * 10, dtype="float32"),
            "int32": np.array([1] * 10, dtype="int32"),
        },
        index=np.arange(10),
    )
    # 对 df 应用 _assert_raw 函数，设置 raw=True，使用指定的引擎进行计算
    df.apply(_assert_raw, axis=axis, engine=engine, raw=True)


def test_apply_axis1(float_frame):
    d = float_frame.index[0]
    # 对 float_frame 应用 np.mean 函数，axis=1，计算结果的索引为 d
    result = float_frame.apply(np.mean, axis=1)[d]
    # 预期结果为 float_frame 中索引为 d 的行的均值
    expected = np.mean(float_frame.xs(d))
    # 断言结果与预期结果是否相等
    assert result == expected


def test_apply_mixed_dtype_corner():
    # 创建一个包含字符串和浮点数的 DataFrame
    df = DataFrame({"A": ["foo"], "B": [1.0]})
    # 对 df 的前 0 行应用 np.mean 函数，axis=1
    result = df[:0].apply(np.mean, axis=1)
    # 预期结果为一个包含 NaN 值的 Series，索引为空的整数索引
    expected = Series(np.nan, index=pd.Index([], dtype="int64"))
    # 断言结果与预期结果是否相等
    tm.assert_series_equal(result, expected)


def test_apply_mixed_dtype_corner_indexing():
    # 创建一个包含字符串和浮点数的 DataFrame
    df = DataFrame({"A": ["foo"], "B": [1.0]})
    # 对 df 应用 lambda 函数，返回每行的 "A" 列，作为结果
    result = df.apply(lambda x: x["A"], axis=1)
    # 预期结果为包含字符串 "foo" 的 Series，索引为 [0]
    expected = Series(["foo"], index=[0])
    # 断言结果与预期结果是否相等
    tm.assert_series_equal(result, expected)

    # 对 df 应用 lambda 函数，返回每行的 "B" 列，作为结果
    result = df.apply(lambda x: x["B"], axis=1)
    # 预期结果为包含浮点数 1.0 的 Series，索引为 [0]
    expected = Series([1.0], index=[0])
    # 断言结果与预期结果是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.parametrize("ax", ["index", "columns"])
@pytest.mark.parametrize(
    "func", [lambda x: x, lambda x: x.mean()], ids=["identity", "mean"]
)
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("axis", [0, 1])
def test_apply_empty_infer_type(ax, func, raw, axis, engine, request):
    # 创建一个包含字符串 "a", "b", "c" 的 DataFrame，轴的类型由 ax 参数指定
    df = DataFrame(**{ax: ["a", "b", "c"]})

    with np.errstate(all="ignore"):
        # 调用 func 函数对一个空的 np.ndarray 应用，以确定其返回类型
        test_res = func(np.array([], dtype="f8"))
        is_reduction = not isinstance(test_res, np.ndarray)

        # 对 df 应用 func 函数，设置 axis 和 raw 参数，使用指定的引擎进行计算
        result = df.apply(func, axis=axis, engine=engine, raw=raw)
        if is_reduction:
            # 如果 func 是一个减少函数，结果为 Series，并且其索引为聚合轴
            agg_axis = df._get_agg_axis(axis)
            assert isinstance(result, Series)
            assert result.index is agg_axis
        else:
            # 否则结果为 DataFrame
            assert isinstance(result, DataFrame)


def test_apply_empty_infer_type_broadcast():
    # 创建一个没有列的 DataFrame，只有索引为 "a", "b", "c"
    no_cols = DataFrame(index=["a", "b", "c"])
    # 使用 apply 方法计算 DataFrame 中每列的均值，使用 lambda 函数 x.mean()，并且以广播方式返回结果
    result = no_cols.apply(lambda x: x.mean(), result_type="broadcast")
    # 断言结果 result 是一个 DataFrame 对象
    assert isinstance(result, DataFrame)
def test_apply_with_args_kwds_add_some(float_frame):
    # 定义一个函数 add_some，接受一个参数 x 和一个默认参数 howmuch，默认为 0，返回 x + howmuch 的结果
    def add_some(x, howmuch=0):
        return x + howmuch

    # 对 float_frame 应用 add_some 函数，howmuch 参数设为 2，返回应用后的结果
    result = float_frame.apply(add_some, howmuch=2)
    # 对 float_frame 应用 lambda 函数，实现每个元素加 2 的操作，作为预期结果
    expected = float_frame.apply(lambda x: x + 2)
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_frame_equal(result, expected)


def test_apply_with_args_kwds_agg_and_add(float_frame):
    # 定义一个函数 agg_and_add，接受一个参数 x 和一个默认参数 howmuch，默认为 0，返回 x 的平均值加上 howmuch 的结果
    def agg_and_add(x, howmuch=0):
        return x.mean() + howmuch

    # 对 float_frame 应用 agg_and_add 函数，howmuch 参数设为 2，返回应用后的结果
    result = float_frame.apply(agg_and_add, howmuch=2)
    # 对 float_frame 应用 lambda 函数，实现每列的平均值加 2 的操作，作为预期结果
    expected = float_frame.apply(lambda x: x.mean() + 2)
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


def test_apply_with_args_kwds_subtract_and_divide(float_frame):
    # 定义一个函数 subtract_and_divide，接受一个参数 x，一个位置参数 sub 和一个默认参数 divide，默认为 1，返回 (x - sub) / divide 的结果
    def subtract_and_divide(x, sub, divide=1):
        return (x - sub) / divide

    # 对 float_frame 应用 subtract_and_divide 函数，args 参数为 (2,)，divide 参数为 2，返回应用后的结果
    result = float_frame.apply(subtract_and_divide, args=(2,), divide=2)
    # 对 float_frame 应用 lambda 函数，实现每个元素减 2 后除以 2 的操作，作为预期结果
    expected = float_frame.apply(lambda x: (x - 2.0) / 2.0)
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_frame_equal(result, expected)


def test_apply_yield_list(float_frame):
    # 对 float_frame 应用 list 函数，返回 float_frame 本身
    result = float_frame.apply(list)
    # 使用测试框架的函数检查 result 是否等于 float_frame
    tm.assert_frame_equal(result, float_frame)


def test_apply_reduce_Series(float_frame):
    # 将 float_frame 的每隔两行的 "A" 列设为 NaN
    float_frame.iloc[::2, float_frame.columns.get_loc("A")] = np.nan
    # 计算 float_frame 每行的均值，作为预期结果
    expected = float_frame.mean(axis=1)
    # 对 float_frame 应用 np.mean 函数，axis 参数设为 1，返回应用后的结果
    result = float_frame.apply(np.mean, axis=1)
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


def test_apply_reduce_to_dict():
    # 创建一个包含数据的 DataFrame，两列 "c0" 和 "c1"，两行 "i0" 和 "i1"
    data = DataFrame([[1, 2], [3, 4]], columns=["c0", "c1"], index=["i0", "i1"])

    # 对 data 应用 dict 函数，axis 参数设为 0，返回列名到 Series 字典的结果
    result = data.apply(dict, axis=0)
    # 创建一个预期结果，包含两个字典，每个字典对应一列的索引和值
    expected = Series([{"i0": 1, "i1": 3}, {"i0": 2, "i1": 4}], index=data.columns)
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)

    # 对 data 应用 dict 函数，axis 参数设为 1，返回行名到 Series 字典的结果
    result = data.apply(dict, axis=1)
    # 创建一个预期结果，包含两个字典，每个字典对应一行的列名和值
    expected = Series([{"c0": 1, "c1": 2}, {"c0": 3, "c1": 4}], index=data.index)
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_series_equal(result, expected)


def test_apply_differently_indexed():
    # 创建一个随机数据的 DataFrame，20 行 10 列
    df = DataFrame(np.random.default_rng(2).standard_normal((20, 10)))

    # 对 df 应用 Series.describe 函数，axis 参数设为 0，返回每列的描述统计信息的 DataFrame
    result = df.apply(Series.describe, axis=0)
    # 创建一个预期结果，包含每列的描述统计信息的 DataFrame
    expected = DataFrame({i: v.describe() for i, v in df.items()}, columns=df.columns)
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_frame_equal(result, expected)

    # 对 df 应用 Series.describe 函数，axis 参数设为 1，返回每行的描述统计信息的 DataFrame
    result = df.apply(Series.describe, axis=1)
    # 创建一个预期结果，包含每行的描述统计信息的 DataFrame
    expected = DataFrame({i: v.describe() for i, v in df.T.items()}, columns=df.index).T
    # 使用测试框架的函数检查 result 是否等于 expected
    tm.assert_frame_equal(result, expected)


def test_apply_bug():
    # 创建一个包含数据的 DataFrame，包含 "a", "market", "position" 三列
    positions = DataFrame(
        [
            [1, "ABC0", 50],
            [1, "YUM0", 20],
            [1, "DEF0", 20],
            [2, "ABC1", 50],
            [2, "YUM1", 20],
            [2, "DEF1", 20],
        ],
        columns=["a", "market", "position"],
    )

    # 定义一个函数 f，返回行中 "market" 列的值
    def f(r):
        return r["market"]

    # 对 positions 应用 f 函数，axis 参数设为 1，返回 Series 对象作为预期结果
    expected = positions.apply(f, axis=1)

    # 重新创建 positions DataFrame，修改第一列为日期类型
    positions = DataFrame(
        [
            [datetime(2013, 1, 1), "ABC0", 50],
            [datetime(2013, 1, 2), "YUM0", 20],
            [datetime(2013, 1, 3), "DEF0", 20],
            [datetime(2013, 1, 4), "ABC1", 50],
            [datetime(2013, 1, 5), "YUM1", 20],
            [datetime(2013, 1, 6), "DEF1", 20],
        ],
        columns=["a", "market", "position"],
    )
    # 对于DataFrame中的每一行，应用函数f，并返回结果Series
    result = positions.apply(f, axis=1)
    # 使用断言验证result和期望的结果Series是否相等
    tm.assert_series_equal(result, expected)
def test_apply_convert_objects():
    # 创建预期的 DataFrame 对象，包含多列数据
    expected = DataFrame(
        {
            "A": [
                "foo",
                "foo",
                "foo",
                "foo",
                "bar",
                "bar",
                "bar",
                "bar",
                "foo",
                "foo",
                "foo",
            ],
            "B": [
                "one",
                "one",
                "one",
                "two",
                "one",
                "one",
                "one",
                "two",
                "two",
                "two",
                "one",
            ],
            "C": [
                "dull",
                "dull",
                "shiny",
                "dull",
                "dull",
                "shiny",
                "shiny",
                "dull",
                "shiny",
                "shiny",
                "shiny",
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    # 应用函数到 DataFrame 的每行，并将结果与预期进行比较
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)


def test_apply_attach_name(float_frame):
    # 将 lambda 函数应用到 DataFrame 列上，返回每列的名称作为结果
    result = float_frame.apply(lambda x: x.name)
    # 创建预期的 Series 对象，包含 DataFrame 列名作为索引和值
    expected = Series(float_frame.columns, index=float_frame.columns)
    tm.assert_series_equal(result, expected)


def test_apply_attach_name_axis1(float_frame):
    # 将 lambda 函数应用到 DataFrame 行上，返回每行的索引作为结果
    result = float_frame.apply(lambda x: x.name, axis=1)
    # 创建预期的 Series 对象，包含 DataFrame 行索引作为索引和值
    expected = Series(float_frame.index, index=float_frame.index)
    tm.assert_series_equal(result, expected)


def test_apply_attach_name_non_reduction(float_frame):
    # 非规约操作：将 lambda 函数应用到每列上，重复列名多次作为结果
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)))
    # 创建预期的 DataFrame 对象，每行都重复列名多次
    expected = DataFrame(
        np.tile(float_frame.columns, (len(float_frame.index), 1)),
        index=float_frame.index,
        columns=float_frame.columns,
    )
    tm.assert_frame_equal(result, expected)


def test_apply_attach_name_non_reduction_axis1(float_frame):
    # 非规约操作：将 lambda 函数应用到每行上，重复行索引多次作为结果
    result = float_frame.apply(lambda x: np.repeat(x.name, len(x)), axis=1)
    # 创建预期的 Series 对象，每行索引重复多次
    expected = Series(
        np.repeat(t[0], len(float_frame.columns)) for t in float_frame.itertuples()
    )
    expected.index = float_frame.index
    tm.assert_series_equal(result, expected)


def test_apply_multi_index():
    # 创建多级索引对象
    index = MultiIndex.from_arrays([["a", "a", "b"], ["c", "d", "d"]])
    # 创建包含多列数据的 DataFrame 对象
    s = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=["col1", "col2"])
    # 应用 lambda 函数到 DataFrame 的每行，返回每行的最小值和最大值构成的 Series 对象
    result = s.apply(lambda x: Series({"min": min(x), "max": max(x)}), 1)
    # 创建预期的 DataFrame 对象，每行包含最小值和最大值列
    expected = DataFrame([[1, 2], [3, 4], [5, 6]], index=index, columns=["min", "max"])
    tm.assert_frame_equal(result, expected, check_like=True)


@pytest.mark.parametrize(
    "df, dicts",
    # 创建一个包含两个列表的列表，每个列表包含两个元素：一个 DataFrame 和一个 Series
    [
        # 第一个列表元素包含：
        [
            # 创建一个 DataFrame，包含两行两列的数据 [["foo", "bar"], ["spam", "eggs"]]
            DataFrame([["foo", "bar"], ["spam", "eggs"]]),
            # 创建一个 Series，包含两个字典 {0: "foo", 1: "spam"} 和 {0: "bar", 1: "eggs"}
            Series([{0: "foo", 1: "spam"}, {0: "bar", 1: "eggs"}]),
        ],
        # 第二个列表元素包含：
        [
            # 创建一个 DataFrame，包含两行两列的数据 [[0, 1], [2, 3]]
            DataFrame([[0, 1], [2, 3]]),
            # 创建一个 Series，包含两个字典 {0: 0, 1: 2} 和 {0: 1, 1: 3}
            Series([{0: 0, 1: 2}, {0: 1, 1: 3}]),
        ],
    ],
def test_apply_dict(df, dicts):
    # GH 8735
    # 定义一个lambda函数fn，用于将DataFrame中的每个元素转换为字典形式
    fn = lambda x: x.to_dict()
    # 对DataFrame应用fn函数，使用"reduce"参数，期望返回Series类型的结果
    reduce_true = df.apply(fn, result_type="reduce")
    # 对DataFrame应用fn函数，使用"expand"参数，期望返回DataFrame类型的结果
    reduce_false = df.apply(fn, result_type="expand")
    # 对DataFrame应用fn函数，未指定result_type参数，默认返回Series类型的结果
    reduce_none = df.apply(fn)
    
    # 使用测试框架tm，断言reduce_true与参数dicts相等
    tm.assert_series_equal(reduce_true, dicts)
    # 使用测试框架tm，断言reduce_false与原始DataFrame df相等
    tm.assert_frame_equal(reduce_false, df)
    # 使用测试框架tm，断言reduce_none与参数dicts相等
    tm.assert_series_equal(reduce_none, dicts)


def test_apply_non_numpy_dtype():
    # GH 12244
    # 创建一个包含日期范围的DataFrame，带有时区信息
    df = DataFrame({"dt": date_range("2015-01-01", periods=3, tz="Europe/Brussels")})
    # 对DataFrame应用一个lambda函数，返回结果与原始DataFrame相等
    result = df.apply(lambda x: x)
    # 使用测试框架tm，断言结果与原始DataFrame相等
    tm.assert_frame_equal(result, df)

    # 对DataFrame应用一个lambda函数，对日期列每个元素加一天
    result = df.apply(lambda x: x + pd.Timedelta("1day"))
    # 创建一个预期的DataFrame，其中日期列每个元素增加一天
    expected = DataFrame(
        {"dt": date_range("2015-01-02", periods=3, tz="Europe/Brussels")}
    )
    # 使用测试框架tm，断言结果与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_apply_non_numpy_dtype_category():
    # 创建一个包含类别数据的DataFrame
    df = DataFrame({"dt": ["a", "b", "c", "a"]}, dtype="category")
    # 对DataFrame应用一个lambda函数，返回结果与原始DataFrame相等
    result = df.apply(lambda x: x)
    # 使用测试框架tm，断言结果与原始DataFrame相等
    tm.assert_frame_equal(result, df)


def test_apply_dup_names_multi_agg():
    # GH 21063
    # 创建一个包含重复列名的DataFrame
    df = DataFrame([[0, 1], [2, 3]], columns=["a", "a"])
    # 对DataFrame应用agg方法，使用"min"作为聚合函数，返回结果DataFrame
    expected = DataFrame([[0, 1]], columns=["a", "a"], index=["min"])
    result = df.agg(["min"])

    # 使用测试框架tm，断言结果DataFrame与预期DataFrame相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("op", ["apply", "agg"])
def test_apply_nested_result_axis_1(op):
    # GH 13820
    # 定义一个用于应用于DataFrame的函数，返回一个列表
    def apply_list(row):
        return [2 * row["A"], 2 * row["C"], 2 * row["B"]]

    # 创建一个全零DataFrame
    df = DataFrame(np.zeros((4, 4)), columns=list("ABCD"))
    # 对DataFrame应用指定的操作（apply或agg），沿着axis=1轴（列）进行计算
    result = getattr(df, op)(apply_list, axis=1)
    # 创建一个预期的Series，其元素为列表的倍数
    expected = Series(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    # 使用测试框架tm，断言结果Series与预期Series相等
    tm.assert_series_equal(result, expected)


def test_apply_noreduction_tzaware_object():
    # https://github.com/pandas-dev/pandas/issues/31505
    # 创建一个包含时区信息的DataFrame
    expected = DataFrame(
        {"foo": [Timestamp("2020", tz="UTC")]}, dtype="datetime64[ns, UTC]"
    )
    # 对DataFrame应用一个lambda函数，返回结果与原始DataFrame相等
    result = expected.apply(lambda x: x)
    # 使用测试框架tm，断言结果与原始DataFrame相等
    tm.assert_frame_equal(result, expected)
    # 对DataFrame应用一个lambda函数，复制每个元素，返回结果与原始DataFrame相等
    result = expected.apply(lambda x: x.copy())
    # 使用测试框架tm，断言结果与原始DataFrame相等
    tm.assert_frame_equal(result, expected)


def test_apply_function_runs_once():
    # https://github.com/pandas-dev/pandas/issues/30815

    # 创建一个包含整数列的DataFrame
    df = DataFrame({"a": [1, 2, 3]})
    names = []  # 保存应用函数的行名

    # 定义一个会修改names的函数
    def reducing_function(row):
        names.append(row.name)

    # 定义一个不会修改names的函数
    def non_reducing_function(row):
        names.append(row.name)
        return row

    # 对每个函数进行测试
    for func in [reducing_function, non_reducing_function]:
        del names[:]  # 清空names列表

        # 对DataFrame应用函数，沿着axis=1轴（行）进行计算
        df.apply(func, axis=1)
        # 使用断言，验证names与DataFrame索引一致
        assert names == list(df.index)


def test_apply_raw_function_runs_once(engine):
    # https://github.com/pandas-dev/pandas/issues/34506
    if engine == "numba":
        pytest.skip("appending to list outside of numba func is not supported")

    # 创建一个包含整数列的DataFrame
    df = DataFrame({"a": [1, 2, 3]})
    values = []  # 保存应用函数的行值

    # 定义一个会修改values的函数
    def reducing_function(row):
        values.extend(row)

    # 定义一个不会修改values的函数
    def non_reducing_function(row):
        values.extend(row)
        return row
    # 遍历函数列表，依次执行每个函数（reducing_function 和 non_reducing_function）
    for func in [reducing_function, non_reducing_function]:
        # 清空 values 列表，准备存储每次函数执行后的结果
        del values[:]

        # 对 DataFrame 应用指定的函数 func，沿着行方向(axis=1)进行操作
        df.apply(func, engine=engine, raw=True, axis=1)
        
        # 断言 values 列表中的内容与 DataFrame 列 'a' 转换为列表后的内容相同
        assert values == list(df.a.to_list())
def test_apply_with_byte_string():
    # GH 34529
    # 创建包含字节字符串的 DataFrame 对象
    df = DataFrame(np.array([b"abcd", b"efgh"]), columns=["col"])
    # 创建期望的 DataFrame 对象，保持对象数据类型不变
    expected = DataFrame(np.array([b"abcd", b"efgh"]), columns=["col"], dtype=object)
    # 对 DataFrame 进行 apply 操作，预期结果是与原始 DataFrame 结构相同，但数据类型为 object
    result = df.apply(lambda x: x.astype("object"))
    # 使用测试工具比较结果和期望的 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize("val", ["asd", 12, None, np.nan])
def test_apply_category_equalness(val):
    # GH 21239
    # 创建包含分类数据的 DataFrame 对象
    df_values = ["asd", None, 12, "asd", "cde", np.nan]
    df = DataFrame({"a": df_values}, dtype="category")
    
    # 对 DataFrame 中的列 'a' 应用 lambda 函数，比较其值与参数 val 是否相等
    result = df.a.apply(lambda x: x == val)
    # 创建预期的 Series 对象，进行相同的比较操作
    expected = Series(
        [np.nan if pd.isnull(x) else x == val for x in df_values], name="a"
    )
    # 使用测试工具比较结果和期望的 Series 是否相等
    tm.assert_series_equal(result, expected)


def test_infer_row_shape():
    # GH 17437
    # 创建一个随机数据的 DataFrame 对象
    df = DataFrame(np.random.default_rng(2).random((10, 2)))
    
    # 对 DataFrame 应用 np.fft.fft 函数，预期结果的形状应该是 (10, 2)
    result = df.apply(np.fft.fft, axis=0).shape
    # 断言结果形状是否符合预期
    assert result == (10, 2)
    
    # 对 DataFrame 应用 np.fft.rfft 函数，预期结果的形状应该是 (6, 2)
    result = df.apply(np.fft.rfft, axis=0).shape
    # 断言结果形状是否符合预期
    assert result == (6, 2)


@pytest.mark.parametrize(
    "ops, by_row, expected",
    [
        # 测试 lambda 函数作为字典操作的应用场景
        ({"a": lambda x: x + 1}, "compat", DataFrame({"a": [2, 3]})),
        ({"a": lambda x: x + 1}, False, DataFrame({"a": [2, 3]})),
        ({"a": lambda x: x.sum()}, "compat", Series({"a": 3})),
        ({"a": lambda x: x.sum()}, False, Series({"a": 3})),
        (
            {"a": ["sum", np.sum, lambda x: x.sum()]},
            "compat",
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            {"a": ["sum", np.sum, lambda x: x.sum()]},
            False,
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        ({"a": lambda x: 1}, "compat", DataFrame({"a": [1, 1]})),
        ({"a": lambda x: 1}, False, Series({"a": 1})),
    ],
)
def test_dictlike_lambda(ops, by_row, expected):
    # GH53601
    # 创建包含整数的 DataFrame 对象
    df = DataFrame({"a": [1, 2]})
    # 对 DataFrame 应用 ops 操作，使用 by_row 参数指定操作方式
    result = df.apply(ops, by_row=by_row)
    # 使用测试工具比较结果和期望的 DataFrame 或 Series 是否相等
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        {"a": lambda x: x + 1},
        {"a": lambda x: x.sum()},
        {"a": ["sum", np.sum, lambda x: x.sum()]},
        {"a": lambda x: 1},
    ],
)
def test_dictlike_lambda_raises(ops):
    # GH53601
    # 创建包含整数的 DataFrame 对象
    df = DataFrame({"a": [1, 2]})
    # 断言应用 ops 操作时，使用 by_row=True 会抛出 ValueError 异常
    with pytest.raises(ValueError, match="by_row=True not allowed"):
        df.apply(ops, by_row=True)


def test_with_dictlike_columns():
    # GH 17602
    # 创建包含整数的 DataFrame 对象
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    # 对 DataFrame 应用 lambda 函数，创建一个包含字典的 Series 对象
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1)
    # 创建预期的 Series 对象，与 apply 操作的结果进行比较
    expected = Series([{"s": 3} for t in df.itertuples()])
    # 使用测试工具比较结果和期望的 Series 是否相等
    tm.assert_series_equal(result, expected)
    # 给 DataFrame 添加名为 "tm" 的时间戳列，包含两个时间戳对象
    df["tm"] = [
        Timestamp("2017-05-01 00:00:00"),  # 第一个时间戳 "2017-05-01 00:00:00"
        Timestamp("2017-05-02 00:00:00"),  # 第二个时间戳 "2017-05-02 00:00:00"
    ]
    
    # 对 DataFrame 进行操作，应用 lambda 函数以计算每行 "a" 列和 "b" 列的和，结果以字典形式返回
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1)
    
    # 使用测试模块 tm 检查 Series 对象 result 是否等于预期的 Series 对象 expected
    tm.assert_series_equal(result, expected)
    
    # 组合一个 Series 对象，计算 DataFrame 中 "a" 列和 "b" 列的和，然后将每个结果包装成字典 {"s": 和}
    result = (df["a"] + df["b"]).apply(lambda x: {"s": x})
    
    # 预期的结果 Series 包含两个元素，每个元素为一个字典 {"s": 3}
    expected = Series([{"s": 3}, {"s": 3}])
    
    # 使用测试模块 tm 检查 Series 对象 result 是否等于预期的 Series 对象 expected
    tm.assert_series_equal(result, expected)
def test_with_dictlike_columns_with_datetime():
    # GH 18775
    # 创建一个空的数据框
    df = DataFrame()
    # 向数据框中添加名为"author"的列
    df["author"] = ["X", "Y", "Z"]
    # 向数据框中添加名为"publisher"的列
    df["publisher"] = ["BBC", "NBC", "N24"]
    # 向数据框中添加名为"date"的列，并将字符串转换为 datetime 对象，以天为先
    df["date"] = pd.to_datetime(
        ["17-10-2010 07:15:30", "13-05-2011 08:20:35", "15-01-2013 09:09:09"],
        dayfirst=True,
    )
    # 对数据框应用 lambda 函数，生成空字典，沿着行的方向
    result = df.apply(lambda x: {}, axis=1)
    # 创建预期结果，为包含空字典的 Series
    expected = Series([{}, {}, {}])
    # 断言结果与预期相等
    tm.assert_series_equal(result, expected)


def test_with_dictlike_columns_with_infer():
    # GH 17602
    # 创建一个包含两行两列的数据框，列名为"a"和"b"
    df = DataFrame([[1, 2], [1, 2]], columns=["a", "b"])
    # 对数据框应用 lambda 函数，生成包含键值对"s"和"a"+"b"的字典，扩展为新列
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1, result_type="expand")
    # 创建预期结果，包含键名"s"和对应值的数据框
    expected = DataFrame({"s": [3, 3]})
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)

    # 向数据框中添加名为"tm"的列，包含时间戳对象
    df["tm"] = [
        Timestamp("2017-05-01 00:00:00"),
        Timestamp("2017-05-02 00:00:00"),
    ]
    # 再次对数据框应用 lambda 函数，生成包含键值对"s"和"a"+"b"的字典，扩展为新列
    result = df.apply(lambda x: {"s": x["a"] + x["b"]}, axis=1, result_type="expand")
    # 断言结果与预期相等
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "ops, by_row, expected",
    [
        # 测试各种操作函数对数据框应用 lambda 函数的结果
        ([lambda x: x + 1], "compat", DataFrame({("a", "<lambda>"): [2, 3]})),
        ([lambda x: x + 1], False, DataFrame({("a", "<lambda>"): [2, 3]})),
        ([lambda x: x.sum()], "compat", DataFrame({"a": [3]}, index=["<lambda>"])),
        ([lambda x: x.sum()], False, DataFrame({"a": [3]}, index=["<lambda>"])),
        (
            ["sum", np.sum, lambda x: x.sum()],
            "compat",
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            ["sum", np.sum, lambda x: x.sum()],
            False,
            DataFrame({"a": [3, 3, 3]}, index=["sum", "sum", "<lambda>"]),
        ),
        (
            [lambda x: x + 1, lambda x: 3],
            "compat",
            DataFrame([[2, 3], [3, 3]], columns=[["a", "a"], ["<lambda>", "<lambda>"]]),
        ),
        (
            [lambda x: 2, lambda x: 3],
            False,
            DataFrame({"a": [2, 3]}, ["<lambda>", "<lambda>"]),
        ),
    ],
)
def test_listlike_lambda(ops, by_row, expected):
    # GH53601
    # 创建一个包含单列"a"的数据框
    df = DataFrame({"a": [1, 2]})
    # 对数据框应用一系列操作函数或 lambda 函数，根据 by_row 参数决定行或列操作
    result = df.apply(ops, by_row=by_row)
    # 断言结果与预期相等
    tm.assert_equal(result, expected)


@pytest.mark.parametrize(
    "ops",
    [
        [lambda x: x + 1],
        [lambda x: x.sum()],
        ["sum", np.sum, lambda x: x.sum()],
        [lambda x: x + 1, lambda x: 3],
    ],
)
def test_listlike_lambda_raises(ops):
    # GH53601
    # 创建一个包含单列"a"的数据框
    df = DataFrame({"a": [1, 2]})
    # 使用 pytest 的断言检查是否引发特定的 ValueError 异常
    with pytest.raises(ValueError, match="by_row=True not allowed"):
        # 对数据框应用一系列操作函数或 lambda 函数，预期引发异常
        df.apply(ops, by_row=True)


def test_with_listlike_columns():
    # GH 17348
    # 创建一个包含三列的数据框，其中"a"列为随机标准正态分布的 Series，"b"列为字符串列表，"ts"列为时间序列
    df = DataFrame(
        {
            "a": Series(np.random.default_rng(2).standard_normal(4)),
            "b": ["a", "list", "of", "words"],
            "ts": date_range("2016-10-01", periods=4, freq="h"),
        }
    )

    # 对数据框中的"a"和"b"列应用 tuple 函数，沿着列的方向
    result = df[["a", "b"]].apply(tuple, axis=1)
    # 创建预期结果，为原数据框中"a"和"b"列元组化的 Series
    expected = Series([t[1:] for t in df[["a", "b"]].itertuples()])
    # 使用测试工具比较两个 Series 是否相等
    tm.assert_series_equal(result, expected)

    # 创建一个新的 Series，其中每行由 DataFrame 列 'a' 和 'ts' 的元组组成
    result = df[["a", "ts"]].apply(tuple, axis=1)

    # 创建预期的 Series，其中每个元素是 DataFrame 列 'a' 和 'ts' 的元组的第二个及其后的元素
    expected = Series([t[1:] for t in df[["a", "ts"]].itertuples()])

    # 使用测试工具比较两个 Series 是否相等
    tm.assert_series_equal(result, expected)
# GH 18919
# 创建一个包含多列的 DataFrame 对象，每列都是 Series 对象
df = DataFrame({"x": Series([["a", "b"], ["q"]]), "y": Series([["z"], ["q", "t"]])})
# 设置 DataFrame 的索引为多级索引
df.index = MultiIndex.from_tuples([("i0", "j0"), ("i1", "j1")])

# 对 DataFrame 应用函数，该函数使用 lambda 表达式对每一行进行处理
result = df.apply(lambda row: [el for el in row["x"] if el in row["y"]], axis=1)
# 期望的结果是一个 Series 对象，包含预期的值
expected = Series([[], ["q"]], index=df.index)
# 使用测试工具函数验证结果是否符合预期
tm.assert_series_equal(result, expected)


# GH 18573
# 创建一个包含多列的 DataFrame 对象，每列包含不同类型的数据
df = DataFrame(
    {
        "number": [1.0, 2.0],
        "string": ["foo", "bar"],
        "datetime": [
            Timestamp("2017-11-29 03:30:00"),
            Timestamp("2017-11-29 03:45:00"),
        ],
    }
)
# 对 DataFrame 应用函数，该函数使用 lambda 表达式返回一个元组
result = df.apply(lambda row: (row.number, row.string), axis=1)
# 期望的结果是一个 Series 对象，包含预期的元组
expected = Series([(t.number, t.string) for t in df.itertuples()])
# 使用测试工具函数验证结果是否符合预期
tm.assert_series_equal(result, expected)


# GH 16353
# 创建一个包含随机数的 DataFrame 对象，列名为 ["A", "B", "C"]
df = DataFrame(
    np.random.default_rng(2).standard_normal((6, 3)), columns=["A", "B", "C"]
)
# 对 DataFrame 应用函数，该函数使用 lambda 表达式返回一个固定列表
result = df.apply(lambda x: [1, 2, 3], axis=1)
# 期望的结果是一个 Series 对象，包含预期的列表
expected = Series([[1, 2, 3] for t in df.itertuples()])
# 使用测试工具函数验证结果是否符合预期
tm.assert_series_equal(result, expected)

# 对 DataFrame 再次应用函数，该函数使用 lambda 表达式返回另一个固定列表
result = df.apply(lambda x: [1, 2], axis=1)
# 期望的结果是一个 Series 对象，包含预期的列表
expected = Series([[1, 2] for t in df.itertuples()])
# 使用测试工具函数验证结果是否符合预期
tm.assert_series_equal(result, expected)


# GH 17970
# 创建一个包含整数的 DataFrame 对象，列名为 ["a"]，索引为 ['a', 'b', 'c']
df = DataFrame({"a": [1, 2, 3]}, index=list("abc"))

# 对 DataFrame 应用函数，该函数使用 lambda 表达式返回一个包含固定值的 NumPy 数组
result = df.apply(lambda row: np.ones(val), axis=1)
# 期望的结果是一个 Series 对象，包含预期的 NumPy 数组
expected = Series([np.ones(val) for t in df.itertuples()], index=df.index)
# 使用测试工具函数验证结果是否符合预期
tm.assert_series_equal(result, expected)


# GH 17892
# 创建一个包含时间戳的 DataFrame 对象，列名为 ["a", "b", "c", "d"]
df = DataFrame(
    {
        "a": [
            Timestamp("2010-02-01"),
            Timestamp("2010-02-04"),
            Timestamp("2010-02-05"),
            Timestamp("2010-02-06"),
        ],
        "b": [9, 5, 4, 3],
        "c": [5, 3, 4, 2],
        "d": [1, 2, 3, 4],
    }
)

# 定义一个函数 fun，返回一个固定的元组 (1, 2)
def fun(x):
    return (1, 2)

# 对 DataFrame 应用函数 fun，该函数返回一个固定的元组 (1, 2)
result = df.apply(fun, axis=1)
# 期望的结果是一个 Series 对象，包含预期的元组
expected = Series([(1, 2) for t in df.itertuples()])
# 使用测试工具函数验证结果是否符合预期
tm.assert_series_equal(result, expected)


# 在参数化测试中使用的测试函数
@pytest.mark.parametrize("lst", [[1, 2, 3], [1, 2]])
def test_consistent_coerce_for_shapes(lst):
    # 我们希望列名不要被传播，只要形状匹配输入形状即可
    # 创建一个包含随机数的 DataFrame 对象，列名为 ["A", "B", "C"]
    df = DataFrame(
        np.random.default_rng(2).standard_normal((4, 3)), columns=["A", "B", "C"]
    )

    # 对 DataFrame 应用函数，该函数使用 lambda 表达式返回参数化的固定列表
    result = df.apply(lambda x: lst, axis=1)
    # 期望的结果是一个 Series 对象，包含预期的列表
    expected = Series([lst for t in df.itertuples()])
    # 使用测试工具函数验证结果是否符合预期
    tm.assert_series_equal(result, expected)


# 测试函数示例
def test_consistent_names(int_frame_const_col):
    # 如果返回一个 Series，我们应该使用结果的索引名称
    df = int_frame_const_col
    # 对 DataFrame 进行 apply 操作，对每行数据应用 lambda 函数生成 Series，指定索引为 ["test", "other", "cols"]
    result = df.apply(
        lambda x: Series([1, 2, 3], index=["test", "other", "cols"]), axis=1
    )
    # 从 int_frame_const_col 中复制 DataFrame，并重命名列名 {"A": "test", "B": "other", "C": "cols"}
    expected = int_frame_const_col.rename(
        columns={"A": "test", "B": "other", "C": "cols"}
    )
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 进行 apply 操作，对每行数据应用 lambda 函数生成 Series，指定索引为 ["test", "other"]
    result = df.apply(lambda x: Series([1, 2], index=["test", "other"]), axis=1)
    # 从 expected 中选择列 "test" 和 "other" 组成新的 DataFrame
    expected = expected[["test", "other"]]
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
# 测试函数，验证应用函数对结果类型的影响
def test_result_type(int_frame_const_col):
    # 无论代码中采用哪条路径，结果类型都应保持一致
    df = int_frame_const_col

    # 对DataFrame应用lambda函数，将结果扩展为新的列
    result = df.apply(lambda x: [1, 2, 3], axis=1, result_type="expand")
    # 创建预期结果DataFrame的副本
    expected = df.copy()
    expected.columns = [0, 1, 2]
    # 使用测试工具比较实际结果和预期结果DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证应用函数对结果类型的影响（较短列表）
def test_result_type_shorter_list(int_frame_const_col):
    # 无论代码中采用哪条路径，结果类型都应保持一致
    df = int_frame_const_col
    # 对DataFrame应用lambda函数，将结果扩展为新的列（较短的列表）
    result = df.apply(lambda x: [1, 2], axis=1, result_type="expand")
    # 创建预期结果DataFrame的副本，仅包含特定列
    expected = df[["A", "B"]].copy()
    expected.columns = [0, 1]
    # 使用测试工具比较实际结果和预期结果DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证应用函数对结果类型的影响（广播模式）
def test_result_type_broadcast(int_frame_const_col, request, engine):
    # 无论代码中采用哪条路径，结果类型都应保持一致
    if engine == "numba":
        # 如果使用numba引擎，标记此测试为预期失败，因为numba不支持列表返回
        mark = pytest.mark.xfail(reason="numba engine doesn't support list return")
        request.node.add_marker(mark)
    df = int_frame_const_col
    # 对DataFrame应用lambda函数，以广播模式扩展结果
    result = df.apply(
        lambda x: [1, 2, 3], axis=1, result_type="broadcast", engine=engine
    )
    # 创建预期结果DataFrame的副本
    expected = df.copy()
    # 使用测试工具比较实际结果和预期结果DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证应用函数对结果类型的影响（使用Series构造函数）
def test_result_type_broadcast_series_func(int_frame_const_col, engine, request):
    # 无论代码中采用哪条路径，结果类型都应保持一致
    if engine == "numba":
        # 如果使用numba引擎，标记此测试为预期失败，因为numba Series构造函数不支持列表数据
        mark = pytest.mark.xfail(
            reason="numba Series constructor only support ndarrays not list data"
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    columns = ["other", "col", "names"]
    # 对DataFrame应用lambda函数，以广播模式扩展结果，并指定列索引
    result = df.apply(
        lambda x: Series([1, 2, 3], index=columns),
        axis=1,
        result_type="broadcast",
        engine=engine,
    )
    # 创建预期结果DataFrame的副本
    expected = df.copy()
    # 使用测试工具比较实际结果和预期结果DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证应用函数对结果类型的影响（返回Series对象）
def test_result_type_series_result(int_frame_const_col, engine, request):
    # 无论代码中采用哪条路径，结果类型都应保持一致
    if engine == "numba":
        # 如果使用numba引擎，标记此测试为预期失败，因为numba Series构造函数不支持列表数据
        mark = pytest.mark.xfail(
            reason="numba Series constructor only support ndarrays not list data"
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    # 对DataFrame应用lambda函数，返回Series对象，并使用原始索引
    result = df.apply(lambda x: Series([1, 2, 3], index=x.index), axis=1, engine=engine)
    # 创建预期结果DataFrame的副本
    expected = df.copy()
    # 使用测试工具比较实际结果和预期结果DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 测试函数，验证应用函数对结果类型的影响（返回Series对象，并指定其他索引）
def test_result_type_series_result_other_index(int_frame_const_col, engine, request):
    # 无论代码中采用哪条路径，结果类型都应保持一致
    if engine == "numba":
        # 如果使用numba引擎，标记此测试为预期失败，因为numba Series构造函数不支持列的列表数据
        mark = pytest.mark.xfail(
            reason="no support in numba Series constructor for list of columns"
        )
        request.node.add_marker(mark)
    df = int_frame_const_col
    columns = ["other", "col", "names"]
    # 对DataFrame应用lambda函数，返回Series对象，并使用指定的列索引
    result = df.apply(lambda x: Series([1, 2, 3], index=columns), axis=1, engine=engine)
    # 复制 DataFrame df，并赋值给 expected 变量
    expected = df.copy()
    # 修改 expected 的列名为指定的 columns 列表中的列名
    expected.columns = columns
    # 使用测试工具 tm.assert_frame_equal 检查 result 和 expected 两个 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)
# 使用 pytest.mark.parametrize 装饰器，为 test_consistency_for_boxed 函数提供多个参数化的测试用例
@pytest.mark.parametrize(
    "box",
    [lambda x: list(x), lambda x: tuple(x), lambda x: np.array(x, dtype="int64")],
    ids=["list", "tuple", "array"],  # 对每个参数化测试用例提供可读性标识符
)
def test_consistency_for_boxed(box, int_frame_const_col):
    # 传递数组或列表不应影响输出形状
    df = int_frame_const_col

    # 测试 lambda 函数应用于 DataFrame 的行，期望结果与使用 itertuples 的 Series 相同
    result = df.apply(lambda x: box([1, 2]), axis=1)
    expected = Series([box([1, 2]) for t in df.itertuples()])
    tm.assert_series_equal(result, expected)

    # 测试 lambda 函数应用于 DataFrame 的行，使用 result_type="expand" 扩展结果
    result = df.apply(lambda x: box([1, 2]), axis=1, result_type="expand")
    expected = int_frame_const_col[["A", "B"]].rename(columns={"A": 0, "B": 1})
    tm.assert_frame_equal(result, expected)


# 测试聚合和转换函数的一致性
def test_agg_transform(axis, float_frame):
    other_axis = 1 if axis in {0, "index"} else 0

    # 忽略所有的 numpy 错误
    with np.errstate(all="ignore"):
        f_abs = np.abs(float_frame)
        f_sqrt = np.sqrt(float_frame)

        # 测试 np.sqrt 函数作为 ufunc 应用于 DataFrame，期望结果与 f_sqrt 相同
        expected = f_sqrt.copy()
        result = float_frame.apply(np.sqrt, axis=axis)
        tm.assert_frame_equal(result, expected)

        # 测试 [np.sqrt] 列表形式应用于 DataFrame，期望结果与 f_sqrt 相同
        result = float_frame.apply([np.sqrt], axis=axis)
        expected = f_sqrt.copy()
        if axis in {0, "index"}:
            expected.columns = MultiIndex.from_product([float_frame.columns, ["sqrt"]])
        else:
            expected.index = MultiIndex.from_product([float_frame.index, ["sqrt"]])
        tm.assert_frame_equal(result, expected)

        # 测试 [np.abs, np.sqrt] 多个函数应用于 DataFrame，期望结果为 f_abs 和 f_sqrt 的组合
        result = float_frame.apply([np.abs, np.sqrt], axis=axis)
        expected = zip_frames([f_abs, f_sqrt], axis=other_axis)
        if axis in {0, "index"}:
            expected.columns = MultiIndex.from_product(
                [float_frame.columns, ["absolute", "sqrt"]]
            )
        else:
            expected.index = MultiIndex.from_product(
                [float_frame.index, ["absolute", "sqrt"]]
            )
        tm.assert_frame_equal(result, expected)


# 演示测试函数
def test_demo():
    # 演示测试函数
    df = DataFrame({"A": range(5), "B": 5})

    # 测试 DataFrame 的聚合函数应用于 ["min", "max"]，期望结果为指定的 DataFrame
    result = df.agg(["min", "max"])
    expected = DataFrame(
        {"A": [0, 4], "B": [5, 5]}, columns=["A", "B"], index=["min", "max"]
    )
    tm.assert_frame_equal(result, expected)


# 使用字典形式的聚合函数演示测试
def test_demo_dict_agg():
    # 演示测试函数
    df = DataFrame({"A": range(5), "B": 5})
    
    # 测试 DataFrame 的字典形式聚合函数应用于 {"A": ["min", "max"], "B": ["sum", "max"]}，期望结果为指定的 DataFrame
    result = df.agg({"A": ["min", "max"], "B": ["sum", "max"]})
    expected = DataFrame(
        {"A": [4.0, 0.0, np.nan], "B": [5.0, np.nan, 25.0]},
        columns=["A", "B"],
        index=["max", "min", "sum"],
    )
    tm.assert_frame_equal(result.reindex_like(expected), expected)


# 使用列名 "name" 的演示测试函数
def test_agg_with_name_as_column_name():
    # GH 36212 - Column name is "name"
    data = {"name": ["foo", "bar"]}
    df = DataFrame(data)

    # 测试 df.agg({"name": "count"})，期望结果为包含 {"name": 2} 的 Series
    result = df.agg({"name": "count"})
    expected = Series({"name": 2})
    tm.assert_series_equal(result, expected)
    # 检查在聚合系列时名称是否仍然保留
    # 使用agg方法对"dataframe"中的"name"列进行聚合，计算数量
    result = df["name"].agg({"name": "count"})
    # 创建预期结果的Series对象，包含{name: 2}，并指定名称为"name"
    expected = Series({"name": 2}, name="name")
    # 使用断言方法检查结果是否与预期相等
    tm.assert_series_equal(result, expected)
def test_agg_multiple_mixed():
    # GH 20909
    # 创建一个包含三列的DataFrame，每列包含不同类型的数据
    mdf = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
        }
    )
    # 期望的DataFrame，包含两行和三列，并指定了行索引
    expected = DataFrame(
        {
            "A": [1, 6],
            "B": [1.0, 6.0],
            "C": ["bar", "foobarbaz"],
        },
        index=["min", "sum"],
    )
    # 对mdf应用agg函数，计算指定的聚合函数（min和sum）
    result = mdf.agg(["min", "sum"])
    # 使用测试框架中的方法验证result是否等于expected
    tm.assert_frame_equal(result, expected)

    # 对mdf中的列'C', 'B', 'A'应用agg函数，计算sum和min，期望结果的行索引按照提供给agg函数的顺序进行排序
    result = mdf[["C", "B", "A"]].agg(["sum", "min"])
    # GH40420: agg函数的结果应具有按提供给agg的参数排序的索引。
    # 从expected中选择'C', 'B', 'A'列并按照sum和min的顺序重新索引
    expected = expected[["C", "B", "A"]].reindex(["sum", "min"])
    # 使用测试框架中的方法验证result是否等于expected
    tm.assert_frame_equal(result, expected)


def test_agg_multiple_mixed_raises():
    # GH 20909
    # 创建一个包含四列的DataFrame，其中一列是日期范围
    mdf = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
            "D": date_range("20130101", periods=3),
        }
    )

    # 对mdf应用agg函数，期望引发TypeError并包含特定消息
    msg = "does not support operation"
    with pytest.raises(TypeError, match=msg):
        mdf.agg(["min", "sum"])

    # 对mdf中的列'D', 'C', 'B', 'A'应用agg函数，期望引发TypeError并包含特定消息
    with pytest.raises(TypeError, match=msg):
        mdf[["D", "C", "B", "A"]].agg(["sum", "min"])


def test_agg_reduce(axis, float_frame):
    # 确定另一个轴的索引，根据axis参数设置为0或'index'选择1，否则选择0
    other_axis = 1 if axis in {0, "index"} else 0
    # 从float_frame的另一个轴（非axis指定的轴）的唯一值中选择前两个，并排序
    name1, name2 = float_frame.axes[other_axis].unique()[:2].sort_values()

    # 创建一个包含所有聚合函数结果的DataFrame，包括mean、max和sum，根据axis参数的不同决定轴方向
    expected = pd.concat(
        [
            float_frame.mean(axis=axis),
            float_frame.max(axis=axis),
            float_frame.sum(axis=axis),
        ],
        axis=1,
    )
    # 设置期望DataFrame的列名为"mean", "max", "sum"，如果axis为0或'index'，则转置expected
    expected.columns = ["mean", "max", "sum"]
    expected = expected.T if axis in {0, "index"} else expected

    # 对float_frame应用agg函数，计算mean、max和sum，根据axis参数确定轴方向
    result = float_frame.agg(["mean", "max", "sum"], axis=axis)
    # 使用测试框架中的方法验证result是否等于expected
    tm.assert_frame_equal(result, expected)

    # 使用字典作为输入，指定name1列为mean，name2列为sum，根据axis参数确定轴方向
    func = {name1: "mean", name2: "sum"}
    # 对float_frame应用agg函数，根据字典func中的指定聚合函数进行计算，根据axis参数确定轴方向
    result = float_frame.agg(func, axis=axis)
    # 创建一个Series，包含根据字典func中指定的聚合函数计算的结果，根据name1和name2索引
    expected = Series(
        [
            float_frame.loc[other_axis][name1].mean(),
            float_frame.loc[other_axis][name2].sum(),
        ],
        index=[name1, name2],
    )
    # 使用测试框架中的方法验证result是否等于expected
    tm.assert_series_equal(result, expected)

    # 使用字典作为输入，指定name1列为mean，name2列为sum的列表，根据axis参数确定轴方向
    func = {name1: ["mean"], name2: ["sum"]}
    # 对float_frame应用agg函数，根据字典func中的指定聚合函数进行计算，根据axis参数确定轴方向
    result = float_frame.agg(func, axis=axis)
    # 创建一个DataFrame，包含根据字典func中指定的聚合函数计算的结果，根据name1和name2列和mean、sum行索引
    expected = DataFrame(
        {
            name1: Series([float_frame.loc[other_axis][name1].mean()], index=["mean"]),
            name2: Series([float_frame.loc[other_axis][name2].sum()], index=["sum"]),
        }
    )
    # 如果axis为1或'columns'，则转置expected
    expected = expected.T if axis in {1, "columns"} else expected
    # 使用测试框架中的方法验证result是否等于expected
    tm.assert_frame_equal(result, expected)

    # 使用字典作为输入，指定name1列为mean、sum，name2列为sum、max的列表，根据axis参数确定轴方向
    func = {name1: ["mean", "sum"], name2: ["sum", "max"]}
    # 对float_frame应用agg函数，根据字典func中的指定聚合函数进行计算，根据axis参数确定轴方向
    result = float_frame.agg(func, axis=axis)
    # 创建一个预期的数据帧 `expected`，通过使用 `pd.concat()` 函数合并以下两个 Series 对象：
    # - 第一个 Series 对象对应 `name1`，包含两个元素：均值和总和，这些值是从 `float_frame` 数据帧的 `other_axis` 轴上取得 `name1` 对应的列进行计算得出的。
    # - 第二个 Series 对象对应 `name2`，包含两个元素：总和和最大值，这些值是从 `float_frame` 数据帧的 `other_axis` 轴上取得 `name2` 对应的列进行计算得出的。
    # 这两个 Series 对象都有自己的索引：第一个 Series 的索引是 ["mean", "sum"]，第二个 Series 的索引是 ["sum", "max"]。
    expected = pd.concat(
        {
            name1: Series(
                [
                    float_frame.loc(other_axis)[name1].mean(),
                    float_frame.loc(other_axis)[name1].sum(),
                ],
                index=["mean", "sum"],
            ),
            name2: Series(
                [
                    float_frame.loc(other_axis)[name2].sum(),
                    float_frame.loc(other_axis)[name2].max(),
                ],
                index=["sum", "max"],
            ),
        },
        axis=1,
    )
    
    # 如果 `axis` 的值是 {1, "columns"}，则将 `expected` 数据帧进行转置，即交换行和列，以便与 `result` 数据帧进行比较。
    # 否则，保持 `expected` 数据帧不变。
    expected = expected.T if axis in {1, "columns"} else expected
    
    # 使用 `tm.assert_frame_equal()` 函数比较 `result` 和 `expected` 两个数据帧，确保它们在内容上是相等的。
    tm.assert_frame_equal(result, expected)
def test_nuiscance_columns():
    # GH 15015
    # 创建一个 DataFrame 对象，包含四列数据：整数、浮点数、字符串和日期
    df = DataFrame(
        {
            "A": [1, 2, 3],
            "B": [1.0, 2.0, 3.0],
            "C": ["foo", "bar", "baz"],
            "D": date_range("20130101", periods=3),
        }
    )

    # 对 DataFrame 应用聚合函数，计算每列的最小值，并返回 Series 对象
    result = df.agg("min")
    # 创建预期结果的 Series 对象，包含每列的最小值
    expected = Series([1, 1.0, "bar", Timestamp("20130101")], index=df.columns)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 对 DataFrame 应用聚合函数，以列表形式指定，计算每列的最小值，并返回 DataFrame 对象
    result = df.agg(["min"])
    # 创建预期结果的 DataFrame 对象，包含一行，每列的最小值
    expected = DataFrame(
        [[1, 1.0, "bar", Timestamp("20130101").as_unit("ns")]],
        index=["min"],
        columns=df.columns,
    )
    # 断言两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)

    # 验证调用不支持的操作时是否引发 TypeError 异常
    msg = "does not support operation"
    with pytest.raises(TypeError, match=msg):
        df.agg("sum")

    # 对部分列应用聚合函数，计算这些列的总和，并返回 Series 对象
    result = df[["A", "B", "C"]].agg("sum")
    # 创建预期结果的 Series 对象，包含部分列的总和
    expected = Series([6, 6.0, "foobarbaz"], index=["A", "B", "C"])
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 验证调用不支持的操作时是否引发 TypeError 异常
    msg = "does not support operation"
    with pytest.raises(TypeError, match=msg):
        df.agg(["sum"])


@pytest.mark.parametrize("how", ["agg", "apply"])
def test_non_callable_aggregates(how):
    # GH 16405
    # 'size' 是 DataFrame 和 Series 的属性，验证其正常工作
    # GH 39116 - 扩展到 apply
    # 创建一个 DataFrame 对象，包含三列数据：整数、浮点数和字符串
    df = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )

    # 使用函数式的聚合方式，计算列 A 的非空元素个数，并返回 Series 对象
    result = getattr(df, how)({"A": "count"})
    # 创建预期结果的 Series 对象，包含列 A 的非空元素个数
    expected = Series({"A": 2})
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 使用非函数式的聚合方式，计算列 A 的元素个数，并返回 Series 对象
    result = getattr(df, how)({"A": "size"})
    # 创建预期结果的 Series 对象，包含列 A 的元素个数
    expected = Series({"A": 3})
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)

    # 混合使用函数和非函数的聚合方式，计算所有列的非空元素个数和元素个数，并返回 DataFrame 对象
    result1 = getattr(df, how)(["count", "size"])
    result2 = getattr(df, how)(
        {"A": ["count", "size"], "B": ["count", "size"], "C": ["count", "size"]}
    )
    # 创建预期结果的 DataFrame 对象，包含每列的非空元素个数和元素个数
    expected = DataFrame(
        {
            "A": {"count": 2, "size": 3},
            "B": {"count": 2, "size": 3},
            "C": {"count": 2, "size": 3},
        }
    )
    # 断言两个 DataFrame 对象是否相等，检查它们是否近似相等
    tm.assert_frame_equal(result1, result2, check_like=True)
    tm.assert_frame_equal(result2, expected, check_like=True)

    # 使用函数式的字符串参数和调用 df.arg() 的效果相同
    result = getattr(df, how)("count")
    # 创建预期结果的 Series 对象，包含每列的元素个数
    expected = df.count()
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("how", ["agg", "apply"])
def test_size_as_str(how, axis):
    # GH 39934
    # 创建一个 DataFrame 对象，包含三列数据：整数、浮点数和字符串
    df = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )
    # 使用字符串属性参数调用和调用 df.arg 效果相同，作用于列
    result = getattr(df, how)("size", axis=axis)
    # 根据 axis 的不同，创建预期结果的 Series 对象，包含每列或每行的元素个数
    if axis in (0, "index"):
        expected = Series(df.shape[0], index=df.columns)
    else:
        expected = Series(df.shape[1], index=df.index)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


def test_agg_listlike_result():
    # GH-29587 user defined function returning list-likes
    # 创建一个包含三列数据的 DataFrame，每列数据类型分别为整数、浮点数和字符串
    df = DataFrame({"A": [2, 2, 3], "B": [1.5, np.nan, 1.5], "C": ["foo", None, "bar"]})
    
    # 定义一个函数 func，用于处理传入的分组列，返回该列去除缺失值后的唯一值列表
    def func(group_col):
        return list(group_col.dropna().unique())
    
    # 对 DataFrame 进行聚合操作，将 func 函数应用到每列上，返回一个 Series，索引为列名，值为 func 处理后的结果
    result = df.agg(func)
    
    # 创建一个期望的 Series，包含每列经过 func 处理后的预期结果，索引为列名
    expected = Series([[2, 3], [1.5], ["foo", "bar"]], index=["A", "B", "C"])
    
    # 使用测试框架 tm 来断言 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
    
    # 对 DataFrame 进行聚合操作，将 func 函数作为整体处理器，返回一个 DataFrame，行为 func 处理结果，列为原 DataFrame 列名
    result = df.agg([func])
    
    # 将预期的 Series 转换为 DataFrame，列名为 "func"，然后转置为与 result 结构一致
    expected = expected.to_frame("func").T
    
    # 使用测试框架 tm 来断言 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((1, 2, 3), {}),
        ((8, 7, 15), {}),
        ((1, 2), {}),
        ((1,), {"b": 2}),
        ((), {"a": 1, "b": 2}),
        ((), {"a": 2, "b": 1}),
        ((), {"a": 1, "b": 2, "c": 3}),
    ],
)
# 定义一个参数化测试函数，用于测试 DataFrame 的聚合函数 agg()
def test_agg_args_kwargs(axis, args, kwargs):
    # 定义一个测试函数 f，对输入的 x 进行求和并根据参数计算返回值
    def f(x, a, b, c=3):
        return x.sum() + (a + b) / c

    # 创建一个 DataFrame 对象
    df = DataFrame([[1, 2], [3, 4]])

    # 根据 axis 的值选择预期的 Series 对象
    if axis == 0:
        expected = Series([5.0, 7.0])
    else:
        expected = Series([4.0, 8.0])

    # 调用 DataFrame 的 agg 方法进行聚合计算
    result = df.agg(f, axis, *args, **kwargs)

    # 使用测试工具函数检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("num_cols", [2, 3, 5])
# 定义一个参数化测试函数，用于测试 DataFrame 的频率是否保持不变
def test_frequency_is_original(num_cols, engine, request):
    # 如果引擎是 "numba"，标记该测试为预期失败
    if engine == "numba":
        mark = pytest.mark.xfail(reason="numba engine only supports numeric indices")
        request.node.add_marker(mark)
    
    # 创建一个 DatetimeIndex 对象
    index = pd.DatetimeIndex(["1950-06-30", "1952-10-24", "1953-05-29"])
    # 复制原始的时间索引
    original = index.copy()
    # 创建一个指定行和列数的 DataFrame 对象
    df = DataFrame(1, index=index, columns=range(num_cols))
    # 应用一个匿名函数到 DataFrame 的每一行
    df.apply(lambda x: x, engine=engine)
    # 断言索引的频率是否与原始索引相同
    assert index.freq == original.freq


def test_apply_datetime_tz_issue(engine, request):
    # GH 29052

    # 如果引擎是 "numba"，标记该测试为预期失败
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support non-numeric indexes"
        )
        request.node.add_marker(mark)

    # 创建一个包含时区信息的 Timestamp 列表
    timestamps = [
        Timestamp("2019-03-15 12:34:31.909000+0000", tz="UTC"),
        Timestamp("2019-03-15 12:34:34.359000+0000", tz="UTC"),
        Timestamp("2019-03-15 12:34:34.660000+0000", tz="UTC"),
    ]
    # 创建一个带有时间戳索引的 DataFrame 对象
    df = DataFrame(data=[0, 1, 2], index=timestamps)
    # 对 DataFrame 应用一个函数，指定使用的引擎
    result = df.apply(lambda x: x.name, axis=1, engine=engine)
    # 创建一个预期的 Series 对象，索引与数据相同
    expected = Series(index=timestamps, data=timestamps)

    # 使用测试工具函数检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("df", [DataFrame({"A": ["a", None], "B": ["c", "d"]})])
@pytest.mark.parametrize("method", ["min", "max", "sum"])
# 定义一个参数化测试函数，用于测试混合列类型的 DataFrame 执行聚合函数是否引发异常
def test_mixed_column_raises(df, method, using_infer_string):
    # 如果方法是 "sum"，设置预期的异常消息
    if method == "sum":
        msg = r'can only concatenate str \(not "int"\) to str|does not support'
    else:
        msg = "not supported between instances of 'str' and 'float'"
    
    # 如果不使用推断字符串类型，则使用 pytest 的断言检查预期的异常消息
    if not using_infer_string:
        with pytest.raises(TypeError, match=msg):
            getattr(df, method)()
    else:
        # 否则，直接调用对应方法
        getattr(df, method)()


@pytest.mark.parametrize("col", [1, 1.0, True, "a", np.nan])
# 定义一个参数化测试函数，用于测试 DataFrame 的 apply 方法对不同类型的列应用函数时返回的 dtype 是否正确
def test_apply_dtype(col):
    # 创建一个包含浮点数和参数化列值的 DataFrame 对象
    df = DataFrame([[1.0, col]], columns=["a", "b"])
    # 对 DataFrame 应用一个 lambda 函数，返回每列的 dtype
    result = df.apply(lambda x: x.dtype)
    # 创建一个预期的 Series 对象，包含 DataFrame 的列 dtype
    expected = df.dtypes

    # 使用测试工具函数检查结果是否与预期相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于测试应用函数时是否会导致 DataFrame 变异
def test_apply_mutating():
    # GH#35462 case where applied func pins a new BlockManager to a row
    # 创建一个包含两列的 DataFrame 对象
    df = DataFrame({"a": range(10), "b": range(10, 20)})
    # 复制原始的 DataFrame 对象
    df_orig = df.copy()

    # 定义一个函数 func，对行进行变异
    def func(row):
        # 获取原始的 BlockManager
        mgr = row._mgr
        # 修改行的 "a" 列
        row.loc["a"] += 1
        # 断言行的 BlockManager 已被更改
        assert row._mgr is not mgr
        return row
    # 复制 DataFrame `df` 到变量 `expected`
    expected = df.copy()
    
    # 将 `expected` DataFrame 中的列 "a" 的每个值增加 1
    expected["a"] += 1
    
    # 对 DataFrame `df` 应用自定义函数 `func`，沿着行的方向（axis=1）进行操作，并将结果存储在 `result` 中
    result = df.apply(func, axis=1)
    
    # 使用测试框架中的函数 `assert_frame_equal` 检查 `result` 和 `expected` 的内容是否相等，用于测试结果的正确性
    tm.assert_frame_equal(result, expected)
    
    # 使用测试框架中的函数 `assert_frame_equal` 检查 `df` 和 `df_orig` 的内容是否相等，用于验证操作的不变性
    tm.assert_frame_equal(df, df_orig)
# GH#35683 get columns correct
def test_apply_empty_list_reduce():
    # 创建包含数据的 DataFrame 对象，包括列名为 'a' 和 'b' 的数据
    df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], columns=["a", "b"])

    # 使用空列表作为应用函数的结果，应用方式为 reduce
    result = df.apply(lambda x: [], result_type="reduce")
    # 预期结果是一个 Series，包含空列表作为值，数据类型为 object
    expected = Series({"a": [], "b": []}, dtype=object)
    # 断言结果与预期是否相等
    tm.assert_series_equal(result, expected)


# GH36189
def test_apply_no_suffix_index(engine, request):
    # 如果引擎为 "numba"，标记为预期失败，因为 numba 引擎不支持列表样式或字典样式的可调用对象
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine doesn't support list-likes/dict-like callables"
        )
        # 将标记添加到测试节点
        request.node.add_marker(mark)
    
    # 创建包含数据的 DataFrame 对象，数据为三行两列，列名为 "A" 和 "B"
    pdf = DataFrame([[4, 9]] * 3, columns=["A", "B"])
    
    # 对 DataFrame 应用函数列表，包括 "sum"、lambda 函数 x.sum()、lambda 函数 x.sum()
    result = pdf.apply(["sum", lambda x: x.sum(), lambda x: x.sum()], engine=engine)
    
    # 预期结果是一个 DataFrame，包含各列的计算结果，行索引为 "sum"、"<lambda>"、"<lambda>"
    expected = DataFrame(
        {"A": [12, 12, 12], "B": [27, 27, 27]}, index=["sum", "<lambda>", "<lambda>"]
    )
    
    # 断言结果 DataFrame 与预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# https://github.com/pandas-dev/pandas/issues/35940
def test_apply_raw_returns_string(engine):
    # 如果引擎为 "numba"，跳过此测试，因为 numba 引擎不支持对象数据类型
    if engine == "numba":
        pytest.skip("No object dtype support in numba")
    
    # 创建包含数据的 DataFrame 对象，包含列名为 "A" 的字符串数据
    df = DataFrame({"A": ["aa", "bbb"]})
    
    # 对 DataFrame 应用函数 lambda 函数 x[0]，在 axis=1 轴上进行原始数据处理
    result = df.apply(lambda x: x[0], engine=engine, axis=1, raw=True)
    
    # 预期结果是一个 Series，包含 ["aa", "bbb"] 的字符串
    expected = Series(["aa", "bbb"])
    
    # 断言结果 Series 与预期 Series 是否相等
    tm.assert_series_equal(result, expected)


# GH40420: the result of .agg should have an index that is sorted
# according to the arguments provided to agg.
def test_aggregation_func_column_order():
    # 创建包含数据的 DataFrame 对象，包括列名为 'att1'、'att2'、'att3' 的数据
    df = DataFrame(
        [
            (1, 0, 0),
            (2, 0, 0),
            (3, 0, 0),
            (4, 5, 4),
            (5, 6, 6),
            (6, 7, 7),
        ],
        columns=("att1", "att2", "att3"),
    )

    # 自定义函数，返回序列的总和除以 2
    def sum_div2(s):
        return s.sum() / 2

    # 聚合函数列表，包括 "sum"、sum_div2 函数、"count"、"min"
    aggs = ["sum", sum_div2, "count", "min"]
    
    # 对 DataFrame 进行聚合操作，根据提供的聚合函数列表进行计算
    result = df.agg(aggs)
    
    # 预期结果是一个 DataFrame，包含各列的聚合计算结果，行索引为 "sum"、"sum_div2"、"count"、"min"
    expected = DataFrame(
        {
            "att1": [21.0, 10.5, 6.0, 1.0],
            "att2": [18.0, 9.0, 6.0, 0.0],
            "att3": [17.0, 8.5, 6.0, 0.0],
        },
        index=["sum", "sum_div2", "count", "min"],
    )
    
    # 断言结果 DataFrame 与预期 DataFrame 是否相等
    tm.assert_frame_equal(result, expected)


# GH 13427
def test_apply_getitem_axis_1(engine, request):
    # 如果引擎为 "numba"，标记为预期失败，因为 numba 引擎不支持重复索引值
    if engine == "numba":
        mark = pytest.mark.xfail(
            reason="numba engine not supporting duplicate index values"
        )
        # 将标记添加到测试节点
        request.node.add_marker(mark)
    
    # 创建包含数据的 DataFrame 对象，包括列名为 'a'、'b' 的数据
    df = DataFrame({"a": [0, 1, 2], "b": [1, 2, 3]})
    
    # 对 DataFrame 进行切片选择，选择 'a' 列的两次，然后应用 lambda 函数求和
    result = df[["a", "a"]].apply(
        lambda x: x.iloc[0] + x.iloc[1], axis=1, engine=engine
    )
    
    # 预期结果是一个 Series，包含 [0, 2, 4] 的数据
    expected = Series([0, 2, 4])
    
    # 断言结果 Series 与预期 Series 是否相等
    tm.assert_series_equal(result, expected)


# GH 43740
def test_nuisance_depr_passes_through_warnings():
    # 定义一个发出警告的函数
    def expected_warning(x):
        warnings.warn("Hello, World!")
        return x.sum()
    # 创建一个包含列"a"的DataFrame，列中包含值[1, 2, 3]
    df = DataFrame({"a": [1, 2, 3]})
    # 使用assert_produces_warning上下文管理器来检查是否会产生UserWarning，并匹配字符串"Hello, World!"
    with tm.assert_produces_warning(UserWarning, match="Hello, World!"):
        # 对DataFrame df 应用agg函数，传入函数expected_warning作为参数
        df.agg([expected_warning])
def test_apply_type():
    # GH 46719
    # 创建一个包含混合数据类型的 DataFrame 对象
    df = DataFrame(
        {"col1": [3, "string", float], "col2": [0.25, datetime(2020, 1, 1), np.nan]},
        index=["a", "b", "c"],
    )

    # 对每一列应用 type 函数，返回每列数据的类型组成的 Series
    result = df.apply(type, axis=0)
    # 预期结果是每列数据类型的 Series
    expected = Series({"col1": Series, "col2": Series})
    # 检查结果是否与预期相符
    tm.assert_series_equal(result, expected)

    # 对每一行应用 type 函数，返回每行数据的类型组成的 Series
    result = df.apply(type, axis=1)
    # 预期结果是每行数据类型的 Series
    expected = Series({"a": Series, "b": Series, "c": Series})
    # 检查结果是否与预期相符
    tm.assert_series_equal(result, expected)


def test_apply_on_empty_dataframe(engine):
    # GH 39111
    # 创建一个包含数据的 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": [3, 0]})
    # 对空 DataFrame 调用 apply 方法，应用 lambda 函数，返回结果 Series
    result = df.head(0).apply(lambda x: max(x["a"], x["b"]), axis=1, engine=engine)
    # 预期结果是一个空的 Series，数据类型为 np.float64
    expected = Series([], dtype=np.float64)
    # 检查结果是否与预期相符
    tm.assert_series_equal(result, expected)


def test_apply_return_list():
    # 创建一个包含数据的 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": [2, 3]})
    # 对 DataFrame 中的每个元素应用 lambda 函数，返回列表形式的结果
    result = df.apply(lambda x: [x.values])
    # 预期结果是一个包含列表的 DataFrame
    expected = DataFrame({"a": [[1, 2]], "b": [[2, 3]]})
    # 检查结果是否与预期相符
    tm.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "test, constant",
    [
        ({"a": [1, 2, 3], "b": [1, 1, 1]}, {"a": [1, 2, 3], "b": [1]}),
        ({"a": [2, 2, 2], "b": [1, 1, 1]}, {"a": [2], "b": [1]}),
    ],
)
def test_unique_agg_type_is_series(test, constant):
    # GH#22558
    # 根据传入的字典创建 DataFrame 对象
    df1 = DataFrame(test)
    # 创建预期的 Series 对象
    expected = Series(data=constant, index=["a", "b"], dtype="object")
    # 定义聚合方式字典
    aggregation = {"a": "unique", "b": "unique"}

    # 对 DataFrame 应用聚合操作
    result = df1.agg(aggregation)

    # 检查结果是否与预期相符
    tm.assert_series_equal(result, expected)


def test_any_apply_keyword_non_zero_axis_regression():
    # https://github.com/pandas-dev/pandas/issues/48656
    # 创建一个包含数据的 DataFrame 对象
    df = DataFrame({"A": [1, 2, 0], "B": [0, 2, 0], "C": [0, 0, 0]})
    # 创建预期的 Series 对象
    expected = Series([True, True, False])
    # 检查是否有任何元素在每行中非零
    tm.assert_series_equal(df.any(axis=1), expected)

    # 对 DataFrame 应用 "any" 函数，沿着 axis=1 方向
    result = df.apply("any", axis=1)
    # 检查结果是否与预期相符
    tm.assert_series_equal(result, expected)

    # 对 DataFrame 应用 "any" 函数，使用非命名参数指定 axis=1
    result = df.apply("any", 1)
    # 检查结果是否与预期相符
    tm.assert_series_equal(result, expected)


def test_agg_mapping_func_deprecated():
    # GH 53325
    # 创建一个包含数据的 DataFrame 对象
    df = DataFrame({"x": [1, 2, 3]})

    # 定义两个函数
    def foo1(x, a=1, c=0):
        return x + a + c

    def foo2(x, b=2, c=0):
        return x + b + c

    # 对 DataFrame 应用 agg 方法，使用单个函数
    result = df.agg(foo1, 0, 3, c=4)
    # 预期结果是 DataFrame 中每个元素加上指定的值
    expected = df + 7
    # 检查结果是否与预期相符
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 应用 agg 方法，使用函数列表
    result = df.agg([foo1, foo2], 0, 3, c=4)
    # 预期结果是包含两列的 DataFrame，每列使用不同的函数计算结果
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]], columns=[["x", "x"], ["foo1", "foo2"]]
    )
    # 检查结果是否与预期相符
    tm.assert_frame_equal(result, expected)

    # 对 DataFrame 应用 agg 方法，使用函数映射
    result = df.agg({"x": foo1}, 0, 3, c=4)
    # 预期结果是一列 DataFrame，应用指定函数计算结果
    expected = DataFrame([2, 3, 4], columns=["x"])
    # 检查结果是否与预期相符
    tm.assert_frame_equal(result, expected)


def test_agg_std():
    # 创建一个包含数据的 DataFrame 对象
    df = DataFrame(np.arange(6).reshape(3, 2), columns=["A", "B"])

    # 对 DataFrame 应用 agg 方法，计算标准差
    result = df.agg(np.std, ddof=1)
    # 预期结果是每列的标准差组成的 Series
    expected = Series({"A": 2.0, "B": 2.0}, dtype=float)
    # 检查结果是否与预期相符
    tm.assert_series_equal(result, expected)

    # 对 DataFrame 应用 agg 方法，计算标准差，使用函数列表形式
    result = df.agg([np.std], ddof=1)
    # 创建一个期望的 DataFrame 对象，包含列名为 "A" 和 "B"，且只有一行数据为 2.0
    expected = DataFrame({"A": 2.0, "B": 2.0}, index=["std"])
    # 使用测试框架中的函数比较 result 和 expected 这两个 DataFrame 对象是否相等
    tm.assert_frame_equal(result, expected)
# 定义一个测试函数，用于测试聚合、分布以及具有非唯一列名的情况
def test_agg_dist_like_and_nonunique_columns():
    # GH#51099：指明此测试与GitHub上的问题编号相关联

    # 创建一个DataFrame对象，包含三列数据：A列有None, 2, 3；B列有1.0, NaN, 3.0；C列有"foo", None, "bar"
    df = DataFrame(
        {"A": [None, 2, 3], "B": [1.0, np.nan, 3.0], "C": ["foo", None, "bar"]}
    )
    
    # 将列名重新设定为"A", "A", "C"，此时"A"列名重复
    df.columns = ["A", "A", "C"]

    # 对DataFrame进行聚合操作，统计"A"列的计数
    result = df.agg({"A": "count"})
    
    # 期望的结果是"A"列的非空值的数量
    expected = df["A"].count()

    # 使用断言函数来比较结果和期望结果是否相等
    tm.assert_series_equal(result, expected)
```