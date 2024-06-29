# `D:\src\scipysrc\pandas\pandas\tests\util\test_assert_series_equal.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据处理和分析
from pandas import (  # 从 Pandas 中导入特定的子模块和类
    Categorical,
    DataFrame,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块


def _assert_series_equal_both(a, b, **kwargs):
    """
    Check that two Series equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    tm.assert_series_equal(a, b, **kwargs)  # 使用 tm.assert_series_equal 检查两个 Series 是否相等
    tm.assert_series_equal(b, a, **kwargs)  # 再次检查，确保检查是对称的


def _assert_not_series_equal(a, b, **kwargs):
    """
    Check that two Series are not equal.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    try:
        tm.assert_series_equal(a, b, **kwargs)  # 尝试使用 tm.assert_series_equal 检查两个 Series 是否相等
        msg = "The two Series were equal when they shouldn't have been"
        pytest.fail(msg=msg)  # 如果相等则抛出断言错误
    except AssertionError:
        pass  # 如果抛出断言错误，则通过


def _assert_not_series_equal_both(a, b, **kwargs):
    """
    Check that two Series are not equal.

    This check is performed commutatively.

    Parameters
    ----------
    a : Series
        The first Series to compare.
    b : Series
        The second Series to compare.
    kwargs : dict
        The arguments passed to `tm.assert_series_equal`.
    """
    _assert_not_series_equal(a, b, **kwargs)  # 调用 _assert_not_series_equal 函数检查两个 Series 不相等
    _assert_not_series_equal(b, a, **kwargs)  # 再次调用，确保检查是对称的


@pytest.mark.parametrize("data", [range(3), list("abc"), list("áàä")])
def test_series_equal(data):
    _assert_series_equal_both(Series(data), Series(data))  # 测试两个相等的 Series


@pytest.mark.parametrize(
    "data1,data2",
    [
        (range(3), range(1, 4)),
        (list("abc"), list("xyz")),
        (list("áàä"), list("éèë")),
        (list("áàä"), list(b"aaa")),
        (range(3), range(4)),
    ],
)
def test_series_not_equal_value_mismatch(data1, data2):
    _assert_not_series_equal_both(Series(data1), Series(data2))  # 测试两个不相等的 Series，数据不匹配


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "float64"},  # 测试 dtype 不匹配的情况
        {"index": [1, 2, 4]},  # 测试 index 不匹配的情况
        {"name": "foo"},  # 测试 name 不匹配的情况
    ],
)
def test_series_not_equal_metadata_mismatch(kwargs):
    data = range(3)
    s1 = Series(data)

    s2 = Series(data, **kwargs)  # 使用不匹配的元数据创建 Series s2
    _assert_not_series_equal_both(s1, s2)  # 测试两个不相等的 Series，元数据不匹配


@pytest.mark.parametrize("data1,data2", [(0.12345, 0.12346), (0.1235, 0.1236)])
@pytest.mark.parametrize("decimals", [0, 1, 2, 3, 5, 10])
def test_less_precise(data1, data2, any_float_dtype, decimals):
    rtol = 10**-decimals  # 设置相对误差容限
    s1 = Series([data1], dtype=any_float_dtype)
    s2 = Series([data2], dtype=any_float_dtype)

    if decimals in (5, 10) or (decimals >= 3 and abs(data1 - data2) >= 0.0005):
        msg = "Series values are different"
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, rtol=rtol)  # 使用 tm.assert_series_equal 检查两个 Series 是否相等
    else:
        # 否则，调用 _assert_series_equal_both 函数，比较 s1 和 s2 两个序列对象的相等性
        _assert_series_equal_both(s1, s2, rtol=rtol)
@pytest.mark.parametrize(
    "s1,s2,msg",
    [
        # Index 指定测试用例参数：s1和s2为Series对象，msg为错误消息
        (
            Series(["l1", "l2"], index=[1, 2]),  # 创建具有非浮点索引的Series对象s1
            Series(["l1", "l2"], index=[1.0, 2.0]),  # 创建具有浮点索引的Series对象s2
            "Series\\.index are different",  # 错误消息描述索引不同
        ),
        # MultiIndex 指定测试用例参数：s1和s2为DataFrame中的某列对象，msg为错误消息
        (
            DataFrame.from_records(
                {"a": [1, 2], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ).c,  # 创建DataFrame并获取其索引为"c"的列对象s1
            DataFrame.from_records(
                {"a": [1.0, 2.0], "b": [2.1, 1.5], "c": ["l1", "l2"]}, index=["a", "b"]
            ).c,  # 创建DataFrame并获取其索引为"c"的列对象s2
            "MultiIndex level \\[0\\] are different",  # 错误消息描述多级索引第一个级别不同
        ),
    ],
)
def test_series_equal_index_dtype(s1, s2, msg, check_index_type):
    kwargs = {"check_index_type": check_index_type}  # 准备传递给assert_series_equal的参数字典

    if check_index_type:
        with pytest.raises(AssertionError, match=msg):  # 检查索引类型时，验证是否抛出断言错误并匹配错误消息
            tm.assert_series_equal(s1, s2, **kwargs)
    else:
        tm.assert_series_equal(s1, s2, **kwargs)  # 否则直接比较两个Series对象


@pytest.mark.parametrize("check_like", [True, False])
def test_series_equal_order_mismatch(check_like):
    s1 = Series([1, 2, 3], index=["a", "b", "c"])  # 创建具有指定索引顺序的Series对象s1
    s2 = Series([3, 2, 1], index=["c", "b", "a"])  # 创建具有不同索引顺序的Series对象s2

    if not check_like:  # 如果不忽略索引顺序
        with pytest.raises(AssertionError, match="Series.index are different"):  # 验证是否抛出断言错误并匹配错误消息
            tm.assert_series_equal(s1, s2, check_like=check_like)
    else:
        _assert_series_equal_both(s1, s2, check_like=check_like)  # 否则调用内部函数比较


@pytest.mark.parametrize("check_index", [True, False])
def test_series_equal_index_mismatch(check_index):
    s1 = Series([1, 2, 3], index=["a", "b", "c"])  # 创建具有指定索引的Series对象s1
    s2 = Series([1, 2, 3], index=["c", "b", "a"])  # 创建具有不同索引的Series对象s2

    if check_index:  # 如果不忽略索引
        with pytest.raises(AssertionError, match="Series.index are different"):  # 验证是否抛出断言错误并匹配错误消息
            tm.assert_series_equal(s1, s2, check_index=check_index)
    else:
        _assert_series_equal_both(s1, s2, check_index=check_index)  # 否则调用内部函数比较


def test_series_invalid_param_combination():
    left = Series(dtype=object)  # 创建具有object类型数据的Series对象left
    right = Series(dtype=object)  # 创建具有object类型数据的Series对象right
    with pytest.raises(
        ValueError, match="check_like must be False if check_index is False"
    ):  # 验证是否抛出值错误并匹配错误消息
        tm.assert_series_equal(left, right, check_index=False, check_like=True)


def test_series_equal_length_mismatch(rtol):
    msg = """Series are different

Series length are different
\\[left\\]:  3, RangeIndex\\(start=0, stop=3, step=1\\)
\\[right\\]: 4, RangeIndex\\(start=0, stop=4, step=1\\)"""

    s1 = Series([1, 2, 3])  # 创建长度为3的Series对象s1
    s2 = Series([1, 2, 3, 4])  # 创建长度为4的Series对象s2

    with pytest.raises(AssertionError, match=msg):  # 验证是否抛出断言错误并匹配多行错误消息
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_numeric_values_mismatch(rtol):
    msg = """Series are different

Series values are different \\(33\\.33333 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[1, 2, 3\\]
\\[right\\]: \\[1, 2, 4\\]"""

    s1 = Series([1, 2, 3])  # 创建具有指定数据的Series对象s1
    s2 = Series([1, 2, 4])  # 创建具有不同数据的Series对象s2

    with pytest.raises(AssertionError, match=msg):  # 验证是否抛出断言错误并匹配多行错误消息
        tm.assert_series_equal(s1, s2, rtol=rtol)
def test_series_equal_categorical_values_mismatch(rtol, using_infer_string):
    # 根据是否使用推断字符串设置不同的错误消息
    if using_infer_string:
        msg = """Series are different

Series values are different \\(66\\.66667 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\['a', 'b', 'c'\\]
Categories \\(3, string\\): \\[a, b, c\\]
\\[right\\]: \\['a', 'c', 'b'\\]
Categories \\(3, string\\): \\[a, b, c\\]"""
    else:
        msg = """Series are different

Series values are different \\(66\\.66667 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\['a', 'b', 'c'\\]
Categories \\(3, object\\): \\['a', 'b', 'c'\\]
\\[right\\]: \\['a', 'c', 'b'\\]
Categories \\(3, object\\): \\['a', 'b', 'c'\\]"""

    # 创建两个Series对象，使用Categorical类型
    s1 = Series(Categorical(["a", "b", "c"]))
    s2 = Series(Categorical(["a", "c", "b"]))

    # 使用pytest检查是否抛出AssertionError，并匹配预期的错误消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_datetime_values_mismatch(rtol):
    # 设置错误消息
    msg = """Series are different

Series values are different \\(100.0 %\\)
\\[index\\]: \\[0, 1, 2\\]
\\[left\\]:  \\[1514764800000000000, 1514851200000000000, 1514937600000000000\\]
\\[right\\]: \\[1549065600000000000, 1549152000000000000, 1549238400000000000\\]"""

    # 创建两个Series对象，使用日期时间数据
    s1 = Series(pd.date_range("2018-01-01", periods=3, freq="D"))
    s2 = Series(pd.date_range("2019-02-02", periods=3, freq="D"))

    # 使用pytest检查是否抛出AssertionError，并匹配预期的错误消息
    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(s1, s2, rtol=rtol)


def test_series_equal_categorical_mismatch(check_categorical, using_infer_string):
    # 根据是否使用推断字符串设置数据类型
    if using_infer_string:
        dtype = "string"
    else:
        dtype = "object"
    # 设置错误消息
    msg = f"""Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  CategoricalDtype\\(categories=\\['a', 'b'\\], ordered=False, \
categories_dtype={dtype}\\)
\\[right\\]: CategoricalDtype\\(categories=\\['a', 'b', 'c'\\], \
ordered=False, categories_dtype={dtype}\\)"""

    # 创建两个Series对象，使用Categorical类型，一个带有更多的类别
    s1 = Series(Categorical(["a", "b"]))
    s2 = Series(Categorical(["a", "b"], categories=list("abc")))

    # 根据条件使用不同的测试方法或抛出AssertionError
    if check_categorical:
        with pytest.raises(AssertionError, match=msg):
            tm.assert_series_equal(s1, s2, check_categorical=check_categorical)
    else:
        _assert_series_equal_both(s1, s2, check_categorical=check_categorical)


def test_assert_series_equal_extension_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    # 创建两个Series对象，使用不同的数据类型
    left = Series(pd.array([1, 2, 3], dtype="Int64"))
    right = left.astype(int)

    # 设置错误消息
    msg = """Attributes of Series are different

Attribute "dtype" are different
\\[left\\]:  Int64
\\[right\\]: int[32|64]"""

    # 检查是否抛出AssertionError，匹配预期的错误消息
    tm.assert_series_equal(left, right, check_dtype=False)

    with pytest.raises(AssertionError, match=msg):
        tm.assert_series_equal(left, right, check_dtype=True)


def test_assert_series_equal_interval_dtype_mismatch():
    # https://github.com/pandas-dev/pandas/issues/32747
    # 创建两个Series对象，使用Interval类型和object类型
    left = Series([pd.Interval(0, 1)], dtype="interval")
    right = left.astype(object)

    # 设置错误消息
    msg = """Attributes of Series are different
def test_series_equal_series_type():
    # 定义一个自定义的Series类
    class MySeries(Series):
        pass

    # 创建几个Series对象
    s1 = Series([1, 2])
    s2 = Series([1, 2])
    s3 = MySeries([1, 2])

    # 比较两个Series对象，忽略其类型是否相同
    tm.assert_series_equal(s1, s2, check_series_type=False)
    tm.assert_series_equal(s1, s2, check_series_type=True)

    # 比较不同类型的Series对象，忽略其类型是否相同
    tm.assert_series_equal(s1, s3, check_series_type=False)
    tm.assert_series_equal(s3, s1, check_series_type=False)

    # 比较不同类型的Series对象，但要求它们的类型相同，引发异常
    with pytest.raises(AssertionError, match="Series classes are different"):
        tm.assert_series_equal(s1, s3, check_series_type=True)

    with pytest.raises(AssertionError, match="Series classes are different"):
        tm.assert_series_equal(s3, s1, check_series_type=True)
    # 创建一个新的索引对象 `idx`，使用序列 `ser` 中的数据
    idx = pd.Index(ser)
    # 使用断言函数 `assert_index_equal` 检查 `idx` 和其深复制的相等性
    tm.assert_index_equal(idx, idx.copy(deep=True))
def test_identical_nested_series_is_equal():
    # GH#22400
    # 创建两个 Pandas Series 对象，包含不同类型的数据
    x = Series(
        [
            0,
            0.0131142231938,
            1.77774652865e-05,
            np.array([0.4722720840328748, 0.4216929783681722]),
        ]
    )
    y = Series(
        [
            0,
            0.0131142231938,
            1.77774652865e-05,
            np.array([0.4722720840328748, 0.4216929783681722]),
        ]
    )
    # 这两个数组应该相等，但是由于嵌套可能导致问题

    # 断言两个 Series 对象相等
    tm.assert_series_equal(x, x)
    # 断言两个 Series 对象严格相等
    tm.assert_series_equal(x, x, check_exact=True)
    # 断言两个 Series 对象相等
    tm.assert_series_equal(x, y)
    # 断言两个 Series 对象严格相等
    tm.assert_series_equal(x, y, check_exact=True)


@pytest.mark.parametrize("dtype", ["datetime64", "timedelta64"])
def test_check_dtype_false_different_reso(dtype):
    # GH 52449
    # 创建一个 Pandas Series 对象，指定数据类型为秒，并转换为毫秒
    ser_s = Series([1000213, 2131232, 21312331]).astype(f"{dtype}[s]")
    ser_ms = ser_s.astype(f"{dtype}[ms]")
    # 使用断言检查 Series 对象是否相等，预期抛出 AssertionError
    with pytest.raises(AssertionError, match="Attributes of Series are different"):
        tm.assert_series_equal(ser_s, ser_ms)
    # 断言两个 Series 对象相等，忽略数据类型检查
    tm.assert_series_equal(ser_ms, ser_s, check_dtype=False)

    # 从 Series 对象中减去另一个 Series 对象，预期抛出 AssertionError
    ser_ms -= Series([1, 1, 1]).astype(f"{dtype}[ms]")
    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(ser_s, ser_ms)

    # 断言两个 Series 对象相等，忽略数据类型检查，预期抛出 AssertionError
    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(ser_s, ser_ms, check_dtype=False)


@pytest.mark.parametrize("dtype", ["Int64", "int64"])
def test_large_unequal_ints(dtype):
    # https://github.com/pandas-dev/pandas/issues/55882
    # 创建两个 Pandas Series 对象，包含大整数数据
    left = Series([1577840521123000], dtype=dtype)
    right = Series([1577840521123543], dtype=dtype)
    # 使用断言检查 Series 对象是否相等，预期抛出 AssertionError
    with pytest.raises(AssertionError, match="Series are different"):
        tm.assert_series_equal(left, right)


@pytest.mark.parametrize("dtype", [None, object])
@pytest.mark.parametrize("check_exact", [True, False])
@pytest.mark.parametrize("val", [3, 3.5])
def test_ea_and_numpy_no_dtype_check(val, check_exact, dtype):
    # GH#56651
    # 创建两个 Pandas Series 对象，一个包含整数或浮点数，另一个包含 NumPy 数组
    left = Series([1, 2, val], dtype=dtype)
    right = Series(pd.array([1, 2, val]))
    # 使用断言检查 Series 对象是否相等，忽略数据类型检查和精确比较的要求
    tm.assert_series_equal(left, right, check_dtype=False, check_exact=check_exact)


def test_assert_series_equal_int_tol():
    # GH#56646
    # 创建两个 Pandas Series 对象，包含整数数据
    left = Series([81, 18, 121, 38, 74, 72, 81, 81, 146, 81, 81, 170, 74, 74])
    right = Series([72, 9, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72])
    # 使用断言检查 Series 对象是否近似相等，设置相对容差
    tm.assert_series_equal(left, right, rtol=1.5)

    # 使用断言检查两个 Series 对象转换为 DataFrame 后是否近似相等，设置相对容差
    tm.assert_frame_equal(left.to_frame(), right.to_frame(), rtol=1.5)

    # 使用断言检查两个 Series 对象转换为扩展数组后是否近似相等，设置相对容差
    tm.assert_extension_array_equal(
        left.astype("Int64").values, right.astype("Int64").values, rtol=1.5
    )
    [
        (
            pd.Index([0, 0.2, 0.4, 0.6, 0.8, 1]),  # 创建一个包含浮点数索引的 Pandas Index 对象
            pd.Index(np.linspace(0, 1, 6)),      # 创建一个包含等差数列的 Pandas Index 对象
        ),
        (
            pd.MultiIndex.from_arrays([[0, 0, 0, 0, 1, 1], [0, 0.2, 0.4, 0.6, 0.8, 1]]),  # 使用数组创建多级索引的 Pandas MultiIndex 对象
            pd.MultiIndex.from_arrays([[0, 0, 0, 0, 1, 1], np.linspace(0, 1, 6)]),    # 使用数组创建多级索引的 Pandas MultiIndex 对象
        ),
        (
            pd.MultiIndex.from_arrays(  # 使用数组创建多级索引的 Pandas MultiIndex 对象
                [["a", "a", "a", "b", "b", "b"], [1, 2, 3, 4, 5, 10000000000001]]
            ),
            pd.MultiIndex.from_arrays(  # 使用数组创建多级索引的 Pandas MultiIndex 对象
                [["a", "a", "a", "b", "b", "b"], [1, 2, 3, 4, 5, 10000000000002]]
            ),
        ),
        pytest.param(  # 使用 pytest.param 包装，用于标记单个测试参数化组合
            pd.Index([1, 2, 3, 4, 5, 10000000000001]),    # 创建包含整数索引的 Pandas Index 对象
            pd.Index([1, 2, 3, 4, 5, 10000000000002]),    # 创建包含整数索引的 Pandas Index 对象
            marks=pytest.mark.xfail(reason="check_exact_index defaults to True"),  # 添加 pytest 标记，表示此测试预期会失败，原因是 check_exact_index 默认为 True
        ),
        pytest.param(  # 使用 pytest.param 包装，用于标记单个测试参数化组合
            pd.MultiIndex.from_arrays(  # 使用数组创建多级索引的 Pandas MultiIndex 对象
                [[0, 0, 0, 0, 1, 1], [1, 2, 3, 4, 5, 10000000000001]]
            ),
            pd.MultiIndex.from_arrays(  # 使用数组创建多级索引的 Pandas MultiIndex 对象
                [[0, 0, 0, 0, 1, 1], [1, 2, 3, 4, 5, 10000000000002]]
            ),
            marks=pytest.mark.xfail(reason="check_exact_index defaults to True"),  # 添加 pytest 标记，表示此测试预期会失败，原因是 check_exact_index 默认为 True
        ),
    ],
# 定义一个函数用于测试，断言两个 Series 对象相等，并检查默认的索引是否完全相同
def test_assert_series_equal_check_exact_index_default(left_idx, right_idx):
    # 创建一个名为 ser1 的 Series 对象，其元素为六个整数 0，使用 left_idx 作为索引
    ser1 = Series(np.zeros(6, dtype=int), left_idx)
    # 创建一个名为 ser2 的 Series 对象，其元素为六个整数 0，使用 right_idx 作为索引
    ser2 = Series(np.zeros(6, dtype=int), right_idx)
    # 使用测试模块中的方法断言 ser1 和 ser2 相等
    tm.assert_series_equal(ser1, ser2)
    # 将 ser1 和 ser2 转换为 DataFrame，并断言这两个 DataFrame 相等
    tm.assert_frame_equal(ser1.to_frame(), ser2.to_frame())
```