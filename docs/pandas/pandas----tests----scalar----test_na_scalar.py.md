# `D:\src\scipysrc\pandas\pandas\tests\scalar\test_na_scalar.py`

```
# 导入日期、时间、时间间隔和 pickle 模块
from datetime import (
    date,
    time,
    timedelta,
)
# 导入 numpy 库，并使用 np 作为别名
import numpy as np
# 导入 pytest 库，用于测试
import pytest
# 导入 pandas 的 NA 值
from pandas._libs.missing import NA
# 导入 pandas 的 is_scalar 函数
from pandas.core.dtypes.common import is_scalar
# 导入 pandas 库，并使用 pd 作为别名
import pandas as pd
# 导入 pandas 的测试工具
import pandas._testing as tm


# 测试 NA 是否为单例
def test_singleton():
    # 断言 NA 和 NA 是同一个对象
    assert NA is NA
    # 创建一个新的 NA 对象，并断言它和之前的 NA 是同一个对象
    new_NA = type(NA)()
    assert new_NA is NA


# 测试 NA 的字符串表示形式
def test_repr():
    # 断言 NA 的 repr 结果为 "<NA>"
    assert repr(NA) == "<NA>"
    # 断言 NA 的 str 结果为 "<NA>"
    assert str(NA) == "<NA>"


# 测试 NA 的格式化输出
def test_format():
    # GH-34740
    # 断言 format 函数处理 NA 的结果为 "<NA>"
    assert format(NA) == "<NA>"
    # 断言 format 函数处理 NA 并右对齐宽度为 10 的结果为 "      <NA>"
    assert format(NA, ">10") == "      <NA>"
    # 断言 format 函数处理 NA 并使用任意格式说明符的结果为 "<NA>"
    assert format(NA, "xxx") == "<NA>"

    # 断言 f-string 处理 NA 的结果为 "<NA>"
    assert f"{NA}" == "<NA>"
    # 断言 f-string 处理 NA 并右对齐宽度为 10 的结果为 "      <NA>"
    assert f"{NA:>10}" == "      <NA>"
    # 断言 f-string 处理 NA 并使用任意格式说明符的结果为 "<NA>"
    assert f"{NA:xxx}" == "<NA>"


# 测试 NA 的真值判断
def test_truthiness():
    msg = "boolean value of NA is ambiguous"

    # 使用 pytest 断言 NA 转换为布尔值会引发 TypeError，且错误信息匹配给定消息
    with pytest.raises(TypeError, match=msg):
        bool(NA)

    # 使用 pytest 断言 对 NA 取反会引发 TypeError，且错误信息匹配给定消息
    with pytest.raises(TypeError, match=msg):
        not NA


# 测试 NA 的可哈希性
def test_hashable():
    # 断言 NA 的哈希值相等
    assert hash(NA) == hash(NA)
    # 创建包含 NA 的字典，并断言获取到的值为 "test"
    d = {NA: "test"}
    assert d[NA] == "test"


# 使用 pytest 的参数化装饰器测试 NA 和各种类型的算术操作
@pytest.mark.parametrize(
    "other", [NA, 1, 1.0, "a", b"a", np.int64(1), np.nan], ids=repr
)
def test_arithmetic_ops(all_arithmetic_functions, other):
    op = all_arithmetic_functions

    if op.__name__ in ("pow", "rpow", "rmod") and isinstance(other, (str, bytes)):
        pytest.skip(reason=f"{op.__name__} with NA and {other} not defined.")
    if op.__name__ in ("divmod", "rdivmod"):
        # 断言 NA 与 other 进行 divmod 或 rdivmod 操作的结果为 (NA, NA)
        assert op(NA, other) is (NA, NA)
    else:
        if op.__name__ == "rpow":
            # 避免特殊情况
            other += 1
        # 断言 NA 和 other 进行当前算术操作的结果为 NA
        assert op(NA, other) is NA


# 使用 pytest 的参数化装饰器测试 NA 和各种类型的比较操作
@pytest.mark.parametrize(
    "other",
    [
        NA,
        1,
        1.0,
        "a",
        b"a",
        np.int64(1),
        np.nan,
        np.bool_(True),
        time(0),
        date(1, 2, 3),
        timedelta(1),
        pd.NaT,
    ],
)
def test_comparison_ops(comparison_op, other):
    # 断言 NA 和 other 进行比较操作的结果为 NA
    assert comparison_op(NA, other) is NA
    # 断言 other 和 NA 进行比较操作的结果为 NA
    assert comparison_op(other, NA) is NA


# 使用 pytest 的参数化装饰器测试 NA 和特殊的幂运算
@pytest.mark.parametrize(
    "value",
    [
        0,
        0.0,
        -0,
        -0.0,
        False,
        np.bool_(False),
        np.int_(0),
        np.float64(0),
        np.int_(-0),
        np.float64(-0),
    ],
)
@pytest.mark.parametrize("asarray", [True, False])
def test_pow_special(value, asarray):
    if asarray:
        value = np.array([value])
    # 对 NA 进行幂运算
    result = NA**value

    if asarray:
        result = result[0]
    else:
        # 对于非数组，无法进行此断言
        assert isinstance(result, type(value))
    # 断言 NA 的幂运算结果为 1
    assert result == 1


# 使用 pytest 的参数化装饰器测试 NA 和特殊的反向幂运算
@pytest.mark.parametrize(
    "value", [1, 1.0, True, np.bool_(True), np.int_(1), np.float64(1)]
)
@pytest.mark.parametrize("asarray", [True, False])
def test_rpow_special(value, asarray):
    if asarray:
        value = np.array([value])
    # 对 value 进行反向幂运算
    result = value**NA

    if asarray:
        result = result[0]
    # 如果 value 不是 np.float64、np.bool_ 或 np.int_ 类型的实例，则执行以下代码块
    elif not isinstance(value, (np.float64, np.bool_, np.int_)):
        # 在 asarray=True 的情况下，无法进行此断言检查
        # 确保 result 是与 value 同一类型的对象
        assert isinstance(result, type(value))

    # 断言 result 的值必须等于 value 的值
    assert result == value
@pytest.mark.parametrize("value", [-1, -1.0, np.int_(-1), np.float64(-1)])
@pytest.mark.parametrize("asarray", [True, False])
def test_rpow_minus_one(value, asarray):
    # 如果 asarray 为 True，则将 value 转换为 NumPy 数组
    if asarray:
        value = np.array([value])
    # 计算 value 的 NA 次方
    result = value**NA

    # 如果之前转换为数组，则取结果的第一个元素
    if asarray:
        result = result[0]

    # 断言结果为 NA
    assert pd.isna(result)


def test_unary_ops():
    # 测试正号操作
    assert +NA is NA
    # 测试负号操作
    assert -NA is NA
    # 测试绝对值操作
    assert abs(NA) is NA
    # 测试按位取反操作
    assert ~NA is NA


def test_logical_and():
    # 测试 NA 与 True 的逻辑与
    assert NA & True is NA
    # 测试 True 与 NA 的逻辑与
    assert True & NA is NA
    # 测试 NA 与 False 的逻辑与
    assert NA & False is False
    # 测试 False 与 NA 的逻辑与
    assert False & NA is False
    # 测试 NA 与 NA 的逻辑与
    assert NA & NA is NA

    # 使用 pytest 检查 NA 与整数 5 的逻辑与操作会引发 TypeError 异常
    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA & 5


def test_logical_or():
    # 测试 NA 或 True 的逻辑或
    assert NA | True is True
    # 测试 True 或 NA 的逻辑或
    assert True | NA is True
    # 测试 NA 或 False 的逻辑或
    assert NA | False is NA
    # 测试 False 或 NA 的逻辑或
    assert False | NA is NA
    # 测试 NA 或 NA 的逻辑或
    assert NA | NA is NA

    # 使用 pytest 检查 NA 或整数 5 的逻辑或操作会引发 TypeError 异常
    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA | 5


def test_logical_xor():
    # 测试 NA 异或 True
    assert NA ^ True is NA
    # 测试 True 异或 NA
    assert True ^ NA is NA
    # 测试 NA 异或 False
    assert NA ^ False is NA
    # 测试 False 异或 NA
    assert False ^ NA is NA
    # 测试 NA 异或 NA
    assert NA ^ NA is NA

    # 使用 pytest 检查 NA 异或整数 5 的操作会引发 TypeError 异常
    msg = "unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        NA ^ 5


def test_logical_not():
    # 测试 NA 的按位取反操作
    assert ~NA is NA


@pytest.mark.parametrize("shape", [(3,), (3, 3), (1, 2, 3)])
def test_arithmetic_ndarray(shape, all_arithmetic_functions):
    op = all_arithmetic_functions
    a = np.zeros(shape)
    # 如果操作名称是 "pow"，则给数组 a 的每个元素加上 5
    if op.__name__ == "pow":
        a += 5
    # 对数组 a 和 NA 进行指定的操作
    result = op(NA, a)
    # 创建一个预期的结果数组，形状与 a 相同，填充值为 NA，数据类型为对象
    expected = np.full(a.shape, NA, dtype=object)
    # 使用测试工具方法检查结果与预期是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_is_scalar():
    # 检查 NA 是否是标量
    assert is_scalar(NA) is True


def test_isna():
    # 检查 pd.isna 是否能正确判断 NA
    assert pd.isna(NA) is True
    # 检查 pd.notna 是否能正确判断 NA
    assert pd.notna(NA) is False


def test_series_isna():
    # 创建包含 NA 的 Pandas Series
    s = pd.Series([1, NA], dtype=object)
    # 创建预期的结果 Series，检查 NA 是否被正确判断为缺失值
    expected = pd.Series([False, True])
    # 使用测试工具方法检查 Series 的 isna() 方法的结果是否符合预期
    tm.assert_series_equal(s.isna(), expected)


def test_ufunc():
    # 测试 np.log() 函数对 NA 的处理
    assert np.log(NA) is NA
    # 测试 np.add() 函数对 NA 和 1 的处理
    assert np.add(NA, 1) is NA
    # 测试 np.divmod() 函数对 NA 和 1 的处理
    result = np.divmod(NA, 1)
    # 检查 np.divmod() 的结果，确保返回的两个值都是 NA
    assert result[0] is NA and result[1] is NA

    # 测试 np.frexp() 函数对 NA 的处理
    result = np.frexp(NA)
    # 检查 np.frexp() 的结果，确保返回的两个值都是 NA
    assert result[0] is NA and result[1] is NA


def test_ufunc_raises():
    # 使用 pytest 检查 np.log.at() 方法对 NA 的操作会引发 ValueError 异常
    msg = "ufunc method 'at'"
    with pytest.raises(ValueError, match=msg):
        np.log.at(NA, 0)


def test_binary_input_not_dunder():
    # 创建一个包含 NA 的数组
    a = np.array([1, 2, 3])
    # 创建预期的结果数组，所有值都是 NA，数据类型为对象
    expected = np.array([NA, NA, NA], dtype=object)
    # 测试 np.logaddexp() 函数对 NA 和数组 a 的处理
    result = np.logaddexp(a, NA)
    # 使用测试工具方法检查结果与预期是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 测试 np.logaddexp() 函数对 NA 和数组 a 的处理（参数顺序颠倒）
    result = np.logaddexp(NA, a)
    # 使用测试工具方法检查结果与预期是否相等
    tm.assert_numpy_array_equal(result, expected)

    # 测试 np.logaddexp() 函数对两个 NA 的处理
    assert np.logaddexp(NA, NA) is NA

    # 测试 np.modf() 函数对两个 NA 的处理
    result = np.modf(NA, NA)
    # 检查 np.modf() 的结果，确保返回的两个值都是 NA
    assert len(result) == 2
    assert all(x is NA for x in result)


def test_divmod_ufunc():
    # 创建一个数组 a
    a = np.array([1, 2, 3])
    # 创建预期的结果数组，所有值都是 NA，数据类型为对象
    expected = np.array([NA, NA, NA], dtype=object)

    # 测试 np.divmod() 函数对数组 a 和 NA 的处理
    result = np.divmod(a, NA)
    # 检查 np.divmod() 的结果类型为元组
    assert isinstance(result, tuple)
    # 遍历结果列表中的每个数组
    for arr in result:
        # 断言数组 arr 与期望值 expected 相等
        tm.assert_numpy_array_equal(arr, expected)
        # 再次断言数组 arr 与期望值 expected 相等

    # 使用 np.divmod 函数对 NA 和 a 进行除法和取余操作，返回结果
    result = np.divmod(NA, a)
    # 遍历结果列表中的每个数组
    for arr in result:
        # 断言数组 arr 与期望值 expected 相等
        tm.assert_numpy_array_equal(arr, expected)
        # 再次断言数组 arr 与期望值 expected 相等
# 定义一个测试函数，用于测试整数哈希冲突在字典中的行为
def test_integer_hash_collision_dict():
    # GH 30013：GitHub 上的 issue 编号，参考链接可以找到更多信息
    result = {NA: "foo", hash(NA): "bar"}

    # 断言字典中 NA 对应的值为 "foo"
    assert result[NA] == "foo"
    # 断言字典中 hash(NA) 对应的值为 "bar"
    assert result[hash(NA)] == "bar"


# 定义一个测试函数，用于测试整数哈希冲突在集合中的行为
def test_integer_hash_collision_set():
    # GH 30013：GitHub 上的 issue 编号，参考链接可以找到更多信息
    result = {NA, hash(NA)}

    # 断言集合的长度为 2
    assert len(result) == 2
    # 断言 NA 在集合中
    assert NA in result
    # 断言 hash(NA) 在集合中
    assert hash(NA) in result


# 定义一个测试函数，用于测试 NA 对象的 pickle 序列化和反序列化
def test_pickle_roundtrip():
    # https://github.com/pandas-dev/pandas/issues/31847：GitHub 上的 issue 编号，参考链接可以找到更多信息
    result = pickle.loads(pickle.dumps(NA))
    # 断言反序列化的结果与原始的 NA 对象是同一个对象
    assert result is NA


# 定义一个测试函数，用于测试 pandas 中 NA 对象的 pickle 序列化和反序列化
def test_pickle_roundtrip_pandas():
    result = tm.round_trip_pickle(NA)
    # 断言反序列化的结果与原始的 NA 对象是同一个对象
    assert result is NA


# 使用 pytest 的参数化装饰器，定义一个测试函数，用于测试不同类型和容器的 pickle 序列化和反序列化
@pytest.mark.parametrize(
    "values, dtype", [([1, 2, NA], "Int64"), (["A", "B", NA], "string")]
)
@pytest.mark.parametrize("as_frame", [True, False])
def test_pickle_roundtrip_containers(as_frame, values, dtype):
    # 创建一个 pandas Series 对象，其中的数据根据给定的 dtype 和 values 创建
    s = pd.Series(pd.array(values, dtype=dtype))
    # 如果 as_frame 为 True，则将 Series 转换为 DataFrame，并命名为 "A"
    if as_frame:
        s = s.to_frame(name="A")
    # 对 Series 或 DataFrame 进行 pickle 序列化和反序列化操作
    result = tm.round_trip_pickle(s)
    # 使用 pandas 测试模块中的 assert_equal 函数，断言序列化和反序列化后的结果与原始对象相等
    tm.assert_equal(result, s)
```