# `D:\src\scipysrc\pandas\pandas\tests\window\test_pairwise.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于编写和运行测试用例

from pandas.compat import IS64  # 导入 IS64 兼容性模块，用于兼容性检查

from pandas import (  # 导入 Pandas 库的多个模块和函数
    DataFrame,  # 数据帧对象
    Index,  # 索引对象
    MultiIndex,  # 多级索引对象
    Series,  # 系列对象
    date_range,  # 日期范围生成函数
)
import pandas._testing as tm  # 导入 Pandas 测试模块，用于测试辅助

from pandas.core.algorithms import safe_sort  # 导入 Pandas 核心算法模块中的安全排序函数


@pytest.fixture(  # 定义 Pytest 的测试夹具，参数化多个数据帧作为输入
    params=[
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1, 0]),  # 创建不同结构的数据帧
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1, 1]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=["C", "C"]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[1.0, 0]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[0.0, 1]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=["C", 1]),
        DataFrame([[2.0, 4.0], [1.0, 2.0], [5.0, 2.0], [8.0, 1.0]], columns=[1, 0.0]),
        DataFrame([[2, 4.0], [1, 2.0], [5, 2.0], [8, 1.0]], columns=[0, 1.0]),
        DataFrame([[2, 4], [1, 2], [5, 2], [8, 1.0]], columns=[1.0, "X"]),
    ]
)
def pairwise_frames(request):  # 定义数据帧对的测试数据生成函数
    """Pairwise frames test_pairwise"""
    return request.param  # 返回参数化的数据帧


@pytest.fixture  # 定义 Pytest 的测试夹具，返回目标数据帧作为输入
def pairwise_target_frame():
    """Pairwise target frame for test_pairwise"""
    return DataFrame([[2, 4], [1, 2], [5, 2], [8, 1]], columns=[0, 1])


@pytest.fixture  # 定义 Pytest 的测试夹具，返回其他数据帧作为输入
def pairwise_other_frame():
    """Pairwise other frame for test_pairwise"""
    return DataFrame(
        [[None, 1, 1], [None, 1, 2], [None, 3, 2], [None, 8, 1]],  # 创建指定结构的数据帧
        columns=["Y", "Z", "X"],
    )


def test_rolling_cov(series):  # 定义滚动协方差测试函数，接受一个系列对象作为输入
    A = series  # 将输入的系列对象赋给变量 A
    B = A + np.random.default_rng(2).standard_normal(len(A))  # 生成随机扰动后的 B 系列

    result = A.rolling(window=50, min_periods=25).cov(B)  # 计算 A 和 B 的滚动协方差
    tm.assert_almost_equal(result.iloc[-1], np.cov(A[-50:], B[-50:])[0, 1])  # 断言结果近似相等


def test_rolling_corr(series):  # 定义滚动相关系数测试函数，接受一个系列对象作为输入
    A = series  # 将输入的系列对象赋给变量 A
    B = A + np.random.default_rng(2).standard_normal(len(A))  # 生成随机扰动后的 B 系列

    result = A.rolling(window=50, min_periods=25).corr(B)  # 计算 A 和 B 的滚动相关系数
    tm.assert_almost_equal(result.iloc[-1], np.corrcoef(A[-50:], B[-50:])[0, 1])  # 断言结果近似相等


def test_rolling_corr_bias_correction():  # 定义滚动相关系数偏差修正测试函数
    # test for correct bias correction
    a = Series(
        np.arange(20, dtype=np.float64),  # 创建一个包含 20 个浮点数的系列对象
        index=date_range("2020-01-01", periods=20)  # 设置日期索引
    )
    b = a.copy()  # 复制系列 a 到 b
    a[:5] = np.nan  # 设置前五个值为 NaN
    b[:10] = np.nan  # 设置前十个值为 NaN

    result = a.rolling(window=len(a), min_periods=1).corr(b)  # 计算 a 和 b 的滚动相关系数
    tm.assert_almost_equal(result.iloc[-1], a.corr(b))  # 断言结果近似相等


@pytest.mark.parametrize("func", ["cov", "corr"])  # 参数化测试函数，测试协方差和相关系数
def test_rolling_pairwise_cov_corr(func, frame):
    result = getattr(frame.rolling(window=10, min_periods=5), func)()  # 使用 getattr 调用指定函数
    result = result.loc[(slice(None), 1), 5]  # 选择结果的特定切片
    result.index = result.index.droplevel(1)  # 删除第二级索引
    expected = getattr(frame[1].rolling(window=10, min_periods=5), func)(frame[5])  # 获取预期结果
    tm.assert_series_equal(result, expected, check_names=False)  # 断言结果系列相等，不检查名称


@pytest.mark.parametrize("method", ["corr", "cov"])  # 参数化测试函数，测试相关系数和协方差
def test_flex_binary_frame(method, frame):
    series = frame[1]  # 选择数据帧的第二列作为系列对象输入

    res = getattr(series.rolling(window=10), method)(frame)  # 使用 getattr 调用指定方法
    res2 = getattr(frame.rolling(window=10), method)(series)  # 使用 getattr 调用指定方法
    # 对 DataFrame 的每一列应用滚动计算，使用 lambda 函数结合 getattr 获取指定方法的结果
    exp = frame.apply(lambda x: getattr(series.rolling(window=10), method)(x))

    # 检查 res 和 exp 是否相等，断言通过则认为测试通过
    tm.assert_frame_equal(res, exp)
    # 再次检查 res2 和 exp 是否相等，确保结果正确性
    tm.assert_frame_equal(res2, exp)

    # 创建一个新的 DataFrame frame2，使用随机数填充，形状与 frame 相同
    frame2 = DataFrame(
        np.random.default_rng(2).standard_normal(frame.shape),
        index=frame.index,
        columns=frame.columns,
    )

    # 对 frame 应用滚动计算的指定方法，传入 frame2 作为参数
    res3 = getattr(frame.rolling(window=10), method)(frame2)
    # 生成期望的结果 DataFrame exp，对 frame 中的每列 k，计算滚动方法的结果
    exp = DataFrame(
        {k: getattr(frame[k].rolling(window=10), method)(frame2[k]) for k in frame}
    )
    # 检查 res3 和 exp 是否相等，以确认计算的正确性
    tm.assert_frame_equal(res3, exp)
@pytest.mark.parametrize("window", range(7))
# 使用 pytest 的 parametrize 标记来定义一个参数化测试，参数为 window 取值范围为 0 到 6
def test_rolling_corr_with_zero_variance(window):
    # 测试函数：检查 rolling 方法计算的相关系数，当输入序列方差为零时的情况
    # 创建一个全零的 Series，长度为 20
    s = Series(np.zeros(20))
    # 创建另一个 Series，包含从 0 到 19 的整数序列
    other = Series(np.arange(20))

    # 断言：检查 rolling 方法计算的相关系数结果是否全部为 NaN
    assert s.rolling(window=window).corr(other=other).isna().all()


def test_corr_sanity():
    # 测试函数：检查 DataFrame 的 rolling 相关系数计算
    # 创建一个 DataFrame 包含特定的数值数组
    df = DataFrame(
        np.array(
            [
                [0.87024726, 0.18505595],
                [0.64355431, 0.3091617],
                [0.92372966, 0.50552513],
                [0.00203756, 0.04520709],
                [0.84780328, 0.33394331],
                [0.78369152, 0.63919667],
            ]
        )
    )

    # 使用 rolling 方法计算第一列与第二列的相关系数，窗口大小为 5，居中计算
    res = df[0].rolling(5, center=True).corr(df[1])
    
    # 断言：检查所有相关系数的绝对值是否不大于 1
    assert all(np.abs(np.nan_to_num(x)) <= 1 for x in res)

    # 创建一个随机生成的 30 行 2 列的 DataFrame
    df = DataFrame(np.random.default_rng(2).random((30, 2)))
    
    # 使用 rolling 方法再次计算第一列与第二列的相关系数，窗口大小为 5，居中计算
    res = df[0].rolling(5, center=True).corr(df[1])
    
    # 断言：检查所有相关系数的绝对值是否不大于 1
    assert all(np.abs(np.nan_to_num(x)) <= 1 for x in res)


def test_rolling_cov_diff_length():
    # 测试函数：检查 rolling 方法计算协方差时处理不同长度序列的情况
    # 创建两个 Series，长度分别为 3 和 2
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    
    # 使用 rolling 方法计算 s1 和 s2 的协方差，窗口大小为 3，最小周期为 2
    result = s1.rolling(window=3, min_periods=2).cov(s2)
    
    # 期望的结果是一个包含 None 和 2.0 的 Series
    expected = Series([None, None, 2.0])
    
    # 使用 assert_series_equal 检查计算结果和期望结果是否一致
    tm.assert_series_equal(result, expected)
    
    # 创建一个包含 None 值的 Series s2a
    s2a = Series([1, None, 3], index=[0, 1, 2])
    
    # 再次使用 rolling 方法计算 s1 和 s2a 的协方差
    result = s1.rolling(window=3, min_periods=2).cov(s2a)
    
    # 使用 assert_series_equal 检查计算结果和期望结果是否一致
    tm.assert_series_equal(result, expected)


def test_rolling_corr_diff_length():
    # 测试函数：检查 rolling 方法计算相关系数时处理不同长度序列的情况
    # 创建两个 Series，长度分别为 3 和 2
    s1 = Series([1, 2, 3], index=[0, 1, 2])
    s2 = Series([1, 3], index=[0, 2])
    
    # 使用 rolling 方法计算 s1 和 s2 的相关系数，窗口大小为 3，最小周期为 2
    result = s1.rolling(window=3, min_periods=2).corr(s2)
    
    # 期望的结果是一个包含 None 和 1.0 的 Series
    expected = Series([None, None, 1.0])
    
    # 使用 assert_series_equal 检查计算结果和期望结果是否一致
    tm.assert_series_equal(result, expected)
    
    # 创建一个包含 None 值的 Series s2a
    s2a = Series([1, None, 3], index=[0, 1, 2])
    
    # 再次使用 rolling 方法计算 s1 和 s2a 的相关系数
    result = s1.rolling(window=3, min_periods=2).corr(s2a)
    
    # 使用 assert_series_equal 检查计算结果和期望结果是否一致
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.rolling(window=10, min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.rolling(window=10, min_periods=5).corr(x, pairwise=True)),
    ],
)
# 使用 pytest 的 parametrize 标记来定义一个参数化测试，参数为 f 取两个 lambda 表达式
def test_rolling_functions_window_non_shrinkage_binary(f):
    # 测试函数：检查 rolling 方法计算协方差和相关系数时返回多重索引的 DataFrame 的情况
    # 创建一个特定的 DataFrame，包含两列名为 "A" 和 "B"
    df = DataFrame(
        [[1, 5], [3, 2], [3, 9], [-1, 0]],
        columns=Index(["A", "B"], name="foo"),
        index=Index(range(4), name="bar"),
    )
    
    # 创建一个期望的 DataFrame，包含与 df 结构相同但数据类型为 float64 的空 DataFrame
    df_expected = DataFrame(
        columns=Index(["A", "B"], name="foo"),
        index=MultiIndex.from_product([df.index, df.columns], names=["bar", "foo"]),
        dtype="float64",
    )
    
    # 调用参数化传入的函数 f，计算 df 的协方差或相关系数的多重索引 DataFrame
    df_result = f(df)
    
    # 使用 assert_frame_equal 检查计算结果和期望结果是否一致
    tm.assert_frame_equal(df_result, df_expected)


@pytest.mark.parametrize(
    "f",
    [
        lambda x: (x.rolling(window=10, min_periods=5).cov(x, pairwise=True)),
        lambda x: (x.rolling(window=10, min_periods=5).corr(x, pairwise=True)),
    ],
)
# 使用 pytest 的 parametrize 标记来定义一个参数化测试，参数为 f 取两个 lambda 表达式
def test_moment_functions_zero_length_pairwise(f):
    # 测试函数：检查处理空 DataFrame 或列的 rolling 方法计算协方差和相关系数的情况
    # 创建一个空的 DataFrame df1
    df1 = DataFrame()
    
    # 创建一个包含一列名为 "a" 的空 DataFrame df2
    df2 = DataFrame(columns=Index(["a"], name="foo"), index=Index([], name="bar"))
    df2["a"] = df2["a"].astype("float64")

    # 创建一个期望的空 DataFrame，其索引为 df1 和 df2 的乘积
    df1_expected = DataFrame(index=MultiIndex.from_product([df1.index, df1.columns]))
    # 创建一个预期的 DataFrame，其索引是 df2 的索引和列的笛卡尔积，列是单一的 "a"，数据类型为 float64
    df2_expected = DataFrame(
        index=MultiIndex.from_product([df2.index, df2.columns], names=["bar", "foo"]),
        columns=Index(["a"], name="foo"),
        dtype="float64",
    )
    
    # 对 df1 应用函数 f，得到处理后的结果 df1_result
    df1_result = f(df1)
    # 使用测试工具 tm 检查 df1_result 是否等于预期的 df1_expected
    tm.assert_frame_equal(df1_result, df1_expected)
    
    # 对 df2 应用函数 f，得到处理后的结果 df2_result
    df2_result = f(df2)
    # 使用测试工具 tm 检查 df2_result 是否等于预期的 df2_expected
    tm.assert_frame_equal(df2_result, df2_expected)
# 定义一个测试类 TestPairwise，用于测试数据框的成对操作
class TestPairwise:
    # 标记测试用例，处理 GitHub 问题 #7738
    @pytest.mark.parametrize("f", [lambda x: x.cov(), lambda x: x.corr()])
    # 定义测试方法 test_no_flex，接受参数 pairwise_frames、pairwise_target_frame 和 f
    def test_no_flex(self, pairwise_frames, pairwise_target_frame, f):
        # 调用参数函数 f 对 pairwise_frames 进行计算
        result = f(pairwise_frames)
        # 断言结果的索引与 pairwise_frames 的列相等
        tm.assert_index_equal(result.index, pairwise_frames.columns)
        # 断言结果的列与 pairwise_frames 的列相等
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        # 使用函数 f 对 pairwise_target_frame 进行计算，赋值给 expected
        expected = f(pairwise_target_frame)
        # 由于结果已排序，只能比较非 NaN 值
        result = result.dropna().values
        expected = expected.dropna().values
        # 断言两个 NumPy 数组相等，不检查数据类型
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    # 标记测试用例，处理成对操作且 pairwise=True 的情况
    @pytest.mark.parametrize(
        "f",
        [
            lambda x: x.expanding().cov(pairwise=True),
            lambda x: x.expanding().corr(pairwise=True),
            lambda x: x.rolling(window=3).cov(pairwise=True),
            lambda x: x.rolling(window=3).corr(pairwise=True),
            lambda x: x.ewm(com=3).cov(pairwise=True),
            lambda x: x.ewm(com=3).corr(pairwise=True),
        ],
    )
    # 定义测试方法 test_pairwise_with_self，接受参数 pairwise_frames、pairwise_target_frame 和 f
    def test_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
        # 调用参数函数 f 对 pairwise_frames 进行计算
        result = f(pairwise_frames)
        # 断言结果的第一层索引与 pairwise_frames 的索引相等，不检查名称
        tm.assert_index_equal(
            result.index.levels[0], pairwise_frames.index, check_names=False
        )
        # 断言结果的第二层索引与 pairwise_frames 列的唯一值排序后相等
        tm.assert_index_equal(
            safe_sort(result.index.levels[1]),
            safe_sort(pairwise_frames.columns.unique()),
        )
        # 断言结果的列与 pairwise_frames 的列相等
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        # 使用函数 f 对 pairwise_target_frame 进行计算，赋值给 expected
        expected = f(pairwise_target_frame)
        # 由于结果已排序，只能比较非 NaN 值
        result = result.dropna().values
        expected = expected.dropna().values
        # 断言两个 NumPy 数组相等，不检查数据类型
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    # 标记测试用例，处理成对操作且 pairwise=False 的情况
    @pytest.mark.parametrize(
        "f",
        [
            lambda x: x.expanding().cov(pairwise=False),
            lambda x: x.expanding().corr(pairwise=False),
            lambda x: x.rolling(window=3).cov(pairwise=False),
            lambda x: x.rolling(window=3).corr(pairwise=False),
            lambda x: x.ewm(com=3).cov(pairwise=False),
            lambda x: x.ewm(com=3).corr(pairwise=False),
        ],
    )
    def test_no_pairwise_with_self(self, pairwise_frames, pairwise_target_frame, f):
        # DataFrame with itself, pairwise=False
        # 使用给定的函数 f 对 pairwise_frames 进行操作，返回结果
        result = f(pairwise_frames)
        # 检查结果的行索引与 pairwise_frames 的行索引是否相等
        tm.assert_index_equal(result.index, pairwise_frames.index)
        # 检查结果的列索引与 pairwise_frames 的列索引是否相等
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        # 使用给定的函数 f 对 pairwise_target_frame 进行操作，返回期望结果
        expected = f(pairwise_target_frame)
        # 由于结果已经排序，只能比较非 NaN 值
        result = result.dropna().values
        expected = expected.dropna().values
        # 断言两个 NumPy 数组是否相等，不检查数据类型
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y, pairwise=True),
            lambda x, y: x.expanding().corr(y, pairwise=True),
            lambda x, y: x.rolling(window=3).cov(y, pairwise=True),
            # TODO: We're missing a flag somewhere in meson
            # 以下为注释部分
            pytest.param(
                lambda x, y: x.rolling(window=3).corr(y, pairwise=True),
                marks=pytest.mark.xfail(
                    not IS64, reason="Precision issues on 32 bit", strict=False
                ),
            ),
            lambda x, y: x.ewm(com=3).cov(y, pairwise=True),
            lambda x, y: x.ewm(com=3).corr(y, pairwise=True),
        ],
    )
    def test_pairwise_with_other(
        self, pairwise_frames, pairwise_target_frame, pairwise_other_frame, f
    ):
        # DataFrame with another DataFrame, pairwise=True
        # 使用给定的函数 f 对 pairwise_frames 和 pairwise_other_frame 进行操作，返回结果
        result = f(pairwise_frames, pairwise_other_frame)
        # 检查结果的第一级索引是否与 pairwise_frames 的索引相等，不检查名称
        tm.assert_index_equal(
            result.index.levels[0], pairwise_frames.index, check_names=False
        )
        # 检查结果的第二级索引是否与 pairwise_other_frame 的列唯一值排序后相等
        tm.assert_index_equal(
            safe_sort(result.index.levels[1]),
            safe_sort(pairwise_other_frame.columns.unique()),
        )
        # 使用给定的函数 f 对 pairwise_target_frame 和 pairwise_other_frame 进行操作，返回期望结果
        expected = f(pairwise_target_frame, pairwise_other_frame)
        # 由于结果已经排序，只能比较非 NaN 值
        result = result.dropna().values
        expected = expected.dropna().values
        # 断言两个 NumPy 数组是否相等，不检查数据类型
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    @pytest.mark.filterwarnings("ignore:RuntimeWarning")
    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y, pairwise=False),
            lambda x, y: x.expanding().corr(y, pairwise=False),
            lambda x, y: x.rolling(window=3).cov(y, pairwise=False),
            lambda x, y: x.rolling(window=3).corr(y, pairwise=False),
            lambda x, y: x.ewm(com=3).cov(y, pairwise=False),
            lambda x, y: x.ewm(com=3).corr(y, pairwise=False),
        ],
    )
    # 测试函数，用于验证不使用成对数据框的情况
    def test_no_pairwise_with_other(self, pairwise_frames, pairwise_other_frame, f):
        # 如果成对数据框的列是唯一的，则调用给定函数f处理数据框
        result = (
            f(pairwise_frames, pairwise_other_frame)
            if pairwise_frames.columns.is_unique
            else None
        )
        if result is not None:
            # 如果结果不为空，计算预期的索引和列，并进行断言比较
            expected_index = pairwise_frames.index.union(pairwise_other_frame.index)
            expected_columns = pairwise_frames.columns.union(
                pairwise_other_frame.columns
            )
            tm.assert_index_equal(result.index, expected_index)
            tm.assert_index_equal(result.columns, expected_columns)
        else:
            # 如果结果为空，测试函数f应该引发值错误异常，包含特定的错误消息
            with pytest.raises(ValueError, match="'arg1' columns are not unique"):
                f(pairwise_frames, pairwise_other_frame)
            with pytest.raises(ValueError, match="'arg2' columns are not unique"):
                f(pairwise_other_frame, pairwise_frames)

    @pytest.mark.parametrize(
        "f",
        [
            lambda x, y: x.expanding().cov(y),
            lambda x, y: x.expanding().corr(y),
            lambda x, y: x.rolling(window=3).cov(y),
            lambda x, y: x.rolling(window=3).corr(y),
            lambda x, y: x.ewm(com=3).cov(y),
            lambda x, y: x.ewm(com=3).corr(y),
        ],
    )
    # 测试函数，用于验证数据框与系列数据的成对操作
    def test_pairwise_with_series(self, pairwise_frames, pairwise_target_frame, f):
        # 调用函数f计算成对数据框与系列数据的结果，并进行索引和列的断言比较
        result = f(pairwise_frames, Series([1, 1, 3, 8]))
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(pairwise_target_frame, Series([1, 1, 3, 8]))
        # 由于结果已经排序，仅比较非 NaN 值
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

        # 交换参数顺序，再次调用函数f计算成对数据框与系列数据的结果，并进行索引和列的断言比较
        result = f(Series([1, 1, 3, 8]), pairwise_frames)
        tm.assert_index_equal(result.index, pairwise_frames.index)
        tm.assert_index_equal(result.columns, pairwise_frames.columns)
        expected = f(Series([1, 1, 3, 8]), pairwise_target_frame)
        # 由于结果已经排序，仅比较非 NaN 值
        result = result.dropna().values
        expected = expected.dropna().values
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    # 测试函数，用于验证相关性计算在内存不足时的行为
    def test_corr_freq_memory_error(self):
        # 创建一个时间序列，计算滚动相关性，并断言结果与预期的时间序列一致
        s = Series(range(5), index=date_range("2020", periods=5))
        result = s.rolling("12h").corr(s)
        expected = Series([np.nan] * 5, index=date_range("2020", periods=5))
        tm.assert_series_equal(result, expected)
    def test_cov_mulittindex(self):
        # GH 34440
        # 创建一个多级索引的列
        columns = MultiIndex.from_product([list("ab"), list("xy"), list("AB")])
        # 创建一个普通的索引
        index = range(3)
        # 创建一个包含数据的 DataFrame，使用 reshape 重塑数组形状
        df = DataFrame(np.arange(24).reshape(3, 8), index=index, columns=columns)

        # 对 DataFrame 使用指数加权移动平均后计算协方差
        result = df.ewm(alpha=0.1).cov()

        # 重新创建一个多级索引的行索引
        index = MultiIndex.from_product([range(3), list("ab"), list("xy"), list("AB")])
        # 重新创建一个多级索引的列
        columns = MultiIndex.from_product([list("ab"), list("xy"), list("AB")])
        # 创建一个期望的 DataFrame，使用 np.vstack 堆叠数组
        expected = DataFrame(
            np.vstack(
                (
                    np.full((8, 8), np.nan),
                    np.full((8, 8), 32.000000),
                    np.full((8, 8), 63.881919),
                )
            ),
            index=index,
            columns=columns,
        )

        # 使用测试框架中的 assert_frame_equal 函数比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_multindex_columns_pairwise_func(self):
        # GH 21157
        # 创建一个多级索引的列，指定列的名称
        columns = MultiIndex.from_arrays([["M", "N"], ["P", "Q"]], names=["a", "b"])
        # 创建一个包含数据的 DataFrame，所有元素为 1
        df = DataFrame(np.ones((5, 2)), columns=columns)
        # 对 DataFrame 使用滚动窗口大小为 3 的滚动相关系数函数
        result = df.rolling(3).corr()

        # 创建一个期望的 DataFrame，所有元素为 NaN
        expected = DataFrame(
            np.nan,
            index=MultiIndex.from_arrays(
                [
                    np.repeat(np.arange(5, dtype=np.int64), 2),
                    ["M", "N"] * 5,
                    ["P", "Q"] * 5,
                ],
                names=[None, "a", "b"],
            ),
            columns=columns,
        )

        # 使用测试框架中的 assert_frame_equal 函数比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
```