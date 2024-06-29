# `D:\src\scipysrc\pandas\pandas\tests\series\test_ufunc.py`

```
# 引入deque模块，用于实现双向队列
from collections import deque
# 引入re模块，用于正则表达式操作
import re
# 引入string模块，提供字符串相关的常量和函数
import string

# 引入NumPy库，并将其命名为np，用于科学计算
import numpy as np
# 引入pytest库，用于编写和运行测试
import pytest

# 引入pandas库，并将其命名为pd，用于数据分析
import pandas as pd
# 引入pandas._testing模块，用于编写Pandas的测试用例
import pandas._testing as tm
# 从pandas.arrays中引入SparseArray类
from pandas.arrays import SparseArray


# pytest fixture，用于提供一个ufunc参数，参数值为np.add或np.logaddexp
@pytest.fixture(params=[np.add, np.logaddexp])
def ufunc(request):
    # 返回参数request.param，即当前的ufunc函数
    # dunder op
    return request.param


# pytest fixture，用于提供一个sparse参数，参数值为True或False，分别对应稀疏和密集模式
@pytest.fixture(params=[True, False], ids=["sparse", "dense"])
def sparse(request):
    # 返回参数request.param，即当前的sparse参数值
    return request.param


# pytest fixture，提供一对随机生成的长度为100的整型数组，大部分为0
@pytest.fixture
def arrays_for_binary_ufunc():
    """
    A pair of random, length-100 integer-dtype arrays, that are mostly 0.
    """
    # 使用NumPy随机数生成器生成长度为100的随机整数数组a1和a2
    a1 = np.random.default_rng(2).integers(0, 10, 100, dtype="int64")
    a2 = np.random.default_rng(2).integers(0, 10, 100, dtype="int64")
    # 将数组a1每隔3个元素置为0
    a1[::3] = 0
    # 将数组a2每隔4个元素置为0
    a2[::4] = 0
    # 返回生成的数组a1和a2作为测试用例
    return a1, a2


# pytest测试用例，参数化测试ufunc为np.positive、np.floor和np.exp
@pytest.mark.parametrize("ufunc", [np.positive, np.floor, np.exp])
def test_unary_ufunc(ufunc, sparse):
    # 测试ufunc(pd.Series) == pd.Series(ufunc)
    # 生成长度为10的随机整数数组arr
    arr = np.random.default_rng(2).integers(0, 10, 10, dtype="int64")
    # 将数组arr每隔2个元素置为0
    arr[::2] = 0
    # 若sparse为True，则使用SparseArray将arr转换为稀疏数组
    if sparse:
        arr = SparseArray(arr, dtype=pd.SparseDtype("int64", 0))

    # 创建长度为10的字母序列作为索引index
    index = list(string.ascii_letters[:10])
    # 设置Series的名称为"name"
    name = "name"
    # 创建Pandas Series对象series，其数据为arr，索引为index，名称为name
    series = pd.Series(arr, index=index, name=name)

    # 对series应用ufunc函数，生成结果result
    result = ufunc(series)
    # 对原始数组arr应用ufunc函数，生成期望结果expected
    expected = pd.Series(ufunc(arr), index=index, name=name)
    # 断言result与expected相等
    tm.assert_series_equal(result, expected)


# pytest测试用例，参数化测试flip为True和False
@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_array(flip, sparse, ufunc, arrays_for_binary_ufunc):
    # 测试ufunc(pd.Series(a), array) == pd.Series(ufunc(a, b))
    a1, a2 = arrays_for_binary_ufunc
    # 若sparse为True，则使用SparseArray将a1和a2转换为稀疏数组
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    # 设置Series的名称为"name"
    name = "name"  # op(pd.Series, array) preserves the name.
    # 创建Pandas Series对象series，其数据为a1，名称为name
    series = pd.Series(a1, name=name)
    # 设置other为数组a2
    other = a2

    # 将a1和a2作为数组参数传递给ufunc，生成数组参数array_args
    array_args = (a1, a2)
    # 将series和other作为Series参数传递给ufunc，生成Series参数series_args
    series_args = (series, other)  # ufunc(series, array)

    # 若flip为True，则颠倒数组参数的顺序
    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)  # ufunc(array, series)

    # 对数组参数应用ufunc函数，生成期望结果expected
    expected = pd.Series(ufunc(*array_args), name=name)
    # 对Series参数应用ufunc函数，生成结果result
    result = ufunc(*series_args)
    # 断言result与expected相等
    tm.assert_series_equal(result, expected)


# pytest测试用例，参数化测试flip为True和False
@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_index(flip, sparse, ufunc, arrays_for_binary_ufunc):
    # 测试
    #   * func(pd.Series(a), pd.Series(b)) == pd.Series(ufunc(a, b))
    #   * ufunc(Index, pd.Series) dispatches to pd.Series (returns a pd.Series)
    a1, a2 = arrays_for_binary_ufunc
    # 若sparse为True，则使用SparseArray将a1和a2转换为稀疏数组
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    # 设置Series的名称为"name"
    name = "name"  # op(pd.Series, array) preserves the name.
    # 创建Pandas Series对象series，其数据为a1，名称为name
    series = pd.Series(a1, name=name)

    # 创建Pandas Index对象other，其数据为a2，名称为name，类型为int64
    other = pd.Index(a2, name=name).astype("int64")

    # 将a1和a2作为数组参数传递给ufunc，生成数组参数array_args
    array_args = (a1, a2)
    # 将series和other作为Series参数传递给ufunc，生成Series参数series_args
    series_args = (series, other)  # ufunc(series, array)
    # 如果 flip 变量为真，反转 array_args 和 series_args 列表的顺序
    if flip:
        array_args = reversed(array_args)
        series_args = reversed(series_args)  # ufunc(array, series)

    # 创建一个期望的 Pandas Series 对象，使用 ufunc 对 array_args 进行计算，指定名称为 name
    expected = pd.Series(ufunc(*array_args), name=name)
    # 使用 ufunc 对 series_args 进行计算，得到结果
    result = ufunc(*series_args)
    # 使用测试模块 tm 来断言 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("shuffle", [True, False], ids=["unaligned", "aligned"])
@pytest.mark.parametrize("flip", [True, False], ids=["flipped", "straight"])
def test_binary_ufunc_with_series(
    flip, shuffle, sparse, ufunc, arrays_for_binary_ufunc
):
    # 测试二元ufunc在Series上的应用
    #   * func(pd.Series(a), pd.Series(b)) == pd.Series(ufunc(a, b))
    #   保证索引对齐
    a1, a2 = arrays_for_binary_ufunc
    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    name = "name"  # op(pd.Series, array)保留名称。
    series = pd.Series(a1, name=name)
    other = pd.Series(a2, name=name)

    idx = np.random.default_rng(2).permutation(len(a1))

    if shuffle:
        other = other.take(idx)
        if flip:
            index = other.align(series)[0].index
        else:
            index = series.align(other)[0].index
    else:
        index = series.index

    array_args = (a1, a2)
    series_args = (series, other)  # ufunc(series, array)

    if flip:
        array_args = tuple(reversed(array_args))
        series_args = tuple(reversed(series_args))  # ufunc(array, series)

    expected = pd.Series(ufunc(*array_args), index=index, name=name)
    result = ufunc(*series_args)
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("flip", [True, False])
def test_binary_ufunc_scalar(ufunc, sparse, flip, arrays_for_binary_ufunc):
    # 测试二元ufunc在标量与Series之间的应用
    #   * ufunc(pd.Series, scalar) == pd.Series(ufunc(array, scalar))
    #   * ufunc(pd.Series, scalar) == ufunc(scalar, pd.Series)
    arr, _ = arrays_for_binary_ufunc
    if sparse:
        arr = SparseArray(arr)
    other = 2
    series = pd.Series(arr, name="name")

    series_args = (series, other)
    array_args = (arr, other)

    if flip:
        series_args = tuple(reversed(series_args))
        array_args = tuple(reversed(array_args))

    expected = pd.Series(ufunc(*array_args), name="name")
    result = ufunc(*series_args)

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("ufunc", [np.divmod])  # TODO: np.modf, np.frexp
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
def test_multiple_output_binary_ufuncs(ufunc, sparse, shuffle, arrays_for_binary_ufunc):
    # 测试多输出的二元ufunc的应用
    #  与二元ufunc_scalar相同的条件适用于具有多个输出的ufunc。

    a1, a2 = arrays_for_binary_ufunc
    # 处理问题 https://github.com/pandas-dev/pandas/issues/26987
    a1[a1 == 0] = 1
    a2[a2 == 0] = 1

    if sparse:
        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0))
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0))

    s1 = pd.Series(a1)
    s2 = pd.Series(a2)

    if shuffle:
        # 在应用ufunc之前确保索引对齐
        s2 = s2.sample(frac=1)

    expected = ufunc(a1, a2)
    assert isinstance(expected, tuple)
    # 调用给定的函数 ufunc，传入参数 s1 和 s2，并获取返回结果
    result = ufunc(s1, s2)
    # 使用断言确保返回的结果是一个元组类型
    assert isinstance(result, tuple)
    # 使用 pandas.testing 模块的 assert_series_equal 函数，
    # 检查 result 的第一个元素是否等于预期的第一个 Series，并保证它们相等
    tm.assert_series_equal(result[0], pd.Series(expected[0]))
    # 检查 result 的第二个元素是否等于预期的第二个 Series，并保证它们相等
    tm.assert_series_equal(result[1], pd.Series(expected[1]))
def test_multiple_output_ufunc(sparse, arrays_for_binary_ufunc):
    # Test that the same conditions from unary input apply to multi-output
    # ufuncs

    arr, _ = arrays_for_binary_ufunc  # 从传入的参数中获取第一个数组，忽略第二个数组

    if sparse:
        arr = SparseArray(arr)  # 如果 sparse 参数为 True，将数组转换为稀疏数组

    series = pd.Series(arr, name="name")  # 使用数组创建一个 pandas Series 对象，设置名称为 "name"
    result = np.modf(series)  # 对 Series 对象中的元素进行按位分解操作，返回浮点数部分和整数部分
    expected = np.modf(arr)  # 对原始数组进行按位分解操作，用于与结果比较

    assert isinstance(result, tuple)  # 断言 result 是一个元组
    assert isinstance(expected, tuple)  # 断言 expected 是一个元组

    # 断言结果的浮点数部分和整数部分与预期一致
    tm.assert_series_equal(result[0], pd.Series(expected[0], name="name"))
    tm.assert_series_equal(result[1], pd.Series(expected[1], name="name"))


def test_binary_ufunc_drops_series_name(ufunc, sparse, arrays_for_binary_ufunc):
    # Drop the names when they differ.

    a1, a2 = arrays_for_binary_ufunc  # 从传入的参数中获取两个数组
    s1 = pd.Series(a1, name="a")  # 使用第一个数组创建一个 pandas Series 对象，设置名称为 "a"
    s2 = pd.Series(a2, name="b")  # 使用第二个数组创建一个 pandas Series 对象，设置名称为 "b"

    result = ufunc(s1, s2)  # 对两个 Series 对象执行给定的二元函数操作
    assert result.name is None  # 断言结果的名称为 None，即删除了输入 Series 对象的名称


def test_object_series_ok():
    class Dummy:
        def __init__(self, value) -> None:
            self.value = value

        def __add__(self, other):
            return self.value + other.value

    arr = np.array([Dummy(0), Dummy(1)])  # 创建一个包含 Dummy 对象的 numpy 数组
    ser = pd.Series(arr)  # 使用数组创建一个 pandas Series 对象
    # 断言执行加法操作后的 Series 对象与预期的 Series 对象相等
    tm.assert_series_equal(np.add(ser, ser), pd.Series(np.add(ser, arr)))
    tm.assert_series_equal(np.add(ser, Dummy(1)), pd.Series(np.add(ser, Dummy(1))))


@pytest.fixture(
    params=[
        pd.array([1, 3, 2], dtype=np.int64),
        pd.array([1, 3, 2], dtype="Int64"),
        pd.array([1, 3, 2], dtype="Float32"),
        pd.array([1, 10, 2], dtype="Sparse[int]"),
        pd.to_datetime(["2000", "2010", "2001"]),
        pd.to_datetime(["2000", "2010", "2001"]).tz_localize("CET"),
        pd.to_datetime(["2000", "2010", "2001"]).to_period(freq="D"),
        pd.to_timedelta(["1 Day", "3 Days", "2 Days"]),
        pd.IntervalIndex([pd.Interval(0, 1), pd.Interval(2, 3), pd.Interval(1, 2)]),
    ],
    ids=lambda x: str(x.dtype),
)
def values_for_np_reduce(request):
    # min/max tests assume that these are monotonic increasing
    # 返回用于 numpy reduce 函数测试的各种数据类型的参数
    return request.param


class TestNumpyReductions:
    # TODO: cases with NAs, axis kwarg for DataFrame
    pass  # 测试 numpy 的归约函数的测试类，暂无具体实现
    # 定义一个测试方法，用于测试 np.multiply.reduce 函数的行为
    def test_multiply(self, values_for_np_reduce, box_with_array, request):
        # 从参数中获取测试所需的数据容器和数值
        box = box_with_array
        values = values_for_np_reduce

        # 断言不会产生任何警告
        with tm.assert_produces_warning(None):
            # 使用数据容器和数值创建对象
            obj = box(values)

        # 如果数据类型是 SparseArray，则标记为预期失败，因为 SparseArray 不支持 'prod' 操作
        if isinstance(values, pd.core.arrays.SparseArray):
            mark = pytest.mark.xfail(reason="SparseArray has no 'prod'")
            request.applymarker(mark)

        # 如果数据类型的基本类型在 "iuf" 中
        if values.dtype.kind in "iuf":
            # 对 obj 中的元素进行乘法操作并且进行累加（reduce）
            result = np.multiply.reduce(obj)
            # 如果数据容器是 DataFrame 类型
            if box is pd.DataFrame:
                # 期望的结果是 obj 中所有元素的乘积，不仅限于数值列
                expected = obj.prod(numeric_only=False)
                # 断言函数执行后的结果和期望的结果是否相等
                tm.assert_series_equal(result, expected)
            # 如果数据容器是 Index 类型
            elif box is pd.Index:
                # Index 类型没有 'prod' 操作，因此计算其元素数组的乘积
                expected = obj._values.prod()
                # 断言函数执行后的结果和期望的结果是否相等
                assert result == expected
            else:
                # 对 obj 中的元素进行乘法操作并累加，期望的结果是所有元素的乘积
                expected = obj.prod()
                # 断言函数执行后的结果和期望的结果是否相等
                assert result == expected
        else:
            # 如果数据类型不在 "iuf" 中，拼接错误信息字符串
            msg = "|".join(
                [
                    "does not support operation",
                    "unsupported operand type",
                    "ufunc 'multiply' cannot use operands",
                ]
            )
            # 断言执行 np.multiply.reduce(obj) 时会引发 TypeError 异常，并且异常信息与预期的错误信息匹配
            with pytest.raises(TypeError, match=msg):
                np.multiply.reduce(obj)

    # 定义一个测试方法，用于测试 np.add.reduce 函数的行为
    def test_add(self, values_for_np_reduce, box_with_array):
        # 从参数中获取测试所需的数据容器和数值
        box = box_with_array
        values = values_for_np_reduce

        # 断言不会产生任何警告
        with tm.assert_produces_warning(None):
            # 使用数据容器和数值创建对象
            obj = box(values)

        # 如果数据类型的基本类型在 "miuf" 中
        if values.dtype.kind in "miuf":
            # 对 obj 中的元素进行加法操作并进行累加（reduce）
            result = np.add.reduce(obj)
            # 如果数据容器是 DataFrame 类型
            if box is pd.DataFrame:
                # 期望的结果是 obj 中所有元素的和，不仅限于数值列
                expected = obj.sum(numeric_only=False)
                # 断言函数执行后的结果和期望的结果是否相等
                tm.assert_series_equal(result, expected)
            # 如果数据容器是 Index 类型
            elif box is pd.Index:
                # Index 类型没有 'sum' 操作，因此计算其元素数组的和
                expected = obj._values.sum()
                # 断言函数执行后的结果和期望的结果是否相等
                assert result == expected
            else:
                # 对 obj 中的元素进行加法操作并累加，期望的结果是所有元素的和
                expected = obj.sum()
                # 断言函数执行后的结果和期望的结果是否相等
                assert result == expected
        else:
            # 如果数据类型不在 "miuf" 中，拼接错误信息字符串
            msg = "|".join(
                [
                    "does not support operation",
                    "unsupported operand type",
                    "ufunc 'add' cannot use operands",
                ]
            )
            # 断言执行 np.add.reduce(obj) 时会引发 TypeError 异常，并且异常信息与预期的错误信息匹配
            with pytest.raises(TypeError, match=msg):
                np.add.reduce(obj)
    # 定义测试方法，用于测试最大值函数
    def test_max(self, values_for_np_reduce, box_with_array):
        # 从参数中获取数组盒子和数值
        box = box_with_array
        values = values_for_np_reduce

        # 初始化类型相同的标志为真
        same_type = True
        # 如果盒子是索引类型且数值的数据类型是整数或浮点数
        if box is pd.Index and values.dtype.kind in ["i", "f"]:
            # 当前情况下，索引会转换为对象，因此我们得到 Python 的整数/浮点数
            same_type = False

        # 禁用任何警告以确保测试的纯净性
        with tm.assert_produces_warning(None):
            # 将数值传递给盒子对象进行初始化
            obj = box(values)

        # 对对象数组使用 NumPy 的最大值函数
        result = np.maximum.reduce(obj)
        # 如果盒子是数据框
        if box is pd.DataFrame:
            # TODO: 处理带有 axis 参数的情况
            # 获取期望的最大值结果，包括非数值列
            expected = obj.max(numeric_only=False)
            # 断言结果与期望相等
            tm.assert_series_equal(result, expected)
        else:
            # 否则，获取期望的第二个数值
            expected = values[1]
            # 断言结果与期望相等
            assert result == expected
            # 如果类型相同标志为真，则检查结果类型与期望类型是否一致
            if same_type:
                assert type(result) == type(expected)

    # 定义测试方法，用于测试最小值函数
    def test_min(self, values_for_np_reduce, box_with_array):
        # 从参数中获取数组盒子和数值
        box = box_with_array
        values = values_for_np_reduce

        # 初始化类型相同的标志为真
        same_type = True
        # 如果盒子是索引类型且数值的数据类型是整数或浮点数
        if box is pd.Index and values.dtype.kind in ["i", "f"]:
            # 当前情况下，索引会转换为对象，因此我们得到 Python 的整数/浮点数
            same_type = False

        # 禁用任何警告以确保测试的纯净性
        with tm.assert_produces_warning(None):
            # 将数值传递给盒子对象进行初始化
            obj = box(values)

        # 对对象数组使用 NumPy 的最小值函数
        result = np.minimum.reduce(obj)
        # 如果盒子是数据框
        if box is pd.DataFrame:
            # 获取期望的最小值结果，包括非数值列
            expected = obj.min(numeric_only=False)
            # 断言结果与期望相等
            tm.assert_series_equal(result, expected)
        else:
            # 否则，获取期望的第一个数值
            expected = values[0]
            # 断言结果与期望相等
            assert result == expected
            # 如果类型相同标志为真，则检查结果类型与期望类型是否一致
            if same_type:
                assert type(result) == type(expected)
@pytest.mark.parametrize("type_", [list, deque, tuple])
# 使用 pytest 的参数化装饰器，测试不同的数据结构类型：list、deque、tuple
def test_binary_ufunc_other_types(type_):
    # 创建一个包含 [1, 2, 3] 的 Pandas Series 对象，命名为 "name"
    a = pd.Series([1, 2, 3], name="name")
    # 使用参数化传入的 type_ 创建另一个数据结构对象，包含 [3, 4, 5]
    b = type_([3, 4, 5])

    # 对 Pandas Series 和数据结构对象进行元素级的加法操作
    result = np.add(a, b)
    # 将 Pandas Series 转换为 NumPy 数组后，再与数据结构对象进行加法操作，返回一个新的 Pandas Series 对象，命名为 "name"
    expected = pd.Series(np.add(a.to_numpy(), b), name="name")
    # 使用 Pandas 提供的测试工具函数，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_object_dtype_ok():
    # 定义一个名为 Thing 的类，具有加法、相等性比较和字符串表示方法
    class Thing:
        def __init__(self, value) -> None:
            self.value = value

        def __add__(self, other):
            other = getattr(other, "value", other)
            return type(self)(self.value + other)

        def __eq__(self, other) -> bool:
            return type(other) is Thing and self.value == other.value

        def __repr__(self) -> str:
            return f"Thing({self.value})"

    # 创建一个包含两个 Thing 对象的 Pandas Series 对象
    s = pd.Series([Thing(1), Thing(2)])
    # 对 Pandas Series 和一个新创建的 Thing 对象进行元素级的加法操作
    result = np.add(s, Thing(1))
    # 创建一个预期的 Pandas Series 对象，与 result 进行比较
    expected = pd.Series([Thing(2), Thing(3)])
    # 使用 Pandas 提供的测试工具函数，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)


def test_outer():
    # 引用 GitHub 上的一个 issue，创建一个包含 [1, 2, 3] 的 Pandas Series 对象
    ser = pd.Series([1, 2, 3])
    # 创建一个包含 [1, 2, 3] 的 NumPy 数组对象
    obj = np.array([1, 2, 3])

    # 使用 pytest 的上下文管理器检查执行以下代码是否会引发 NotImplementedError 异常，匹配异常消息为空字符串
    with pytest.raises(NotImplementedError, match=""):
        # 对 Pandas Series 和 NumPy 数组执行 np.subtract.outer 操作
        np.subtract.outer(ser, obj)


def test_np_matmul():
    # 引用 GitHub 上的一个 issue，创建一个包含 [[-1, 1, 10]] 的 Pandas DataFrame 对象
    df1 = pd.DataFrame(data=[[-1, 1, 10]])
    # 创建一个包含 [-1, 1, 10] 的 Pandas DataFrame 对象
    df2 = pd.DataFrame(data=[-1, 1, 10])
    # 创建一个预期的 Pandas DataFrame 对象，包含 [[102]]
    expected = pd.DataFrame(data=[102])

    # 对两个 Pandas DataFrame 对象执行矩阵乘法操作
    result = np.matmul(df1, df2)
    # 使用 Pandas 提供的测试工具函数，比较 result 和 expected 是否相等
    tm.assert_frame_equal(expected, result)


@pytest.mark.parametrize("box", [pd.Index, pd.Series])
# 使用 pytest 的参数化装饰器，测试不同的 Pandas 对象类型：pd.Index 和 pd.Series
def test_np_matmul_1D(box):
    # 对两个包含 [1, 2] 的 Pandas 对象（box）执行矩阵乘法操作
    result = np.matmul(box([1, 2]), box([2, 3]))
    # 断言结果是否等于 8
    assert result == 8
    # 断言结果的类型是否为 np.int64
    assert isinstance(result, np.int64)


def test_array_ufuncs_for_many_arguments():
    # 引用 GitHub 上的一个 issue，定义一个将三个参数相加的函数
    def add3(x, y, z):
        return x + y + z

    # 使用 np.frompyfunc 创建一个能够处理三个参数的通用函数对象
    ufunc = np.frompyfunc(add3, 3, 1)
    # 创建一个包含 [1, 2] 的 Pandas Series 对象
    ser = pd.Series([1, 2])

    # 使用通用函数对象对 Pandas Series 执行三个参数的相加操作，得到一个新的 Pandas Series 对象
    result = ufunc(ser, ser, 1)
    # 创建一个预期的 Pandas Series 对象，包含 [3, 5]
    expected = pd.Series([3, 5], dtype=object)
    # 使用 Pandas 提供的测试工具函数，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)

    # 创建一个包含 [[1, 2]] 的 Pandas DataFrame 对象
    df = pd.DataFrame([[1, 2]])

    # 准备一条错误消息，用于检查是否引发了 NotImplementedError 异常，消息内容包含一定的文本匹配
    msg = (
        "Cannot apply ufunc <ufunc 'add3 (vectorized)'> "
        "to mixed DataFrame and Series inputs."
    )
    # 使用 pytest 的上下文管理器检查执行以下代码是否会引发 NotImplementedError 异常，匹配错误消息变量 msg
    with pytest.raises(NotImplementedError, match=re.escape(msg)):
        # 使用通用函数对象对 Pandas Series 和 Pandas DataFrame 执行三个参数的相加操作
        ufunc(ser, ser, df)


@pytest.mark.xfail(reason="see https://github.com/pandas-dev/pandas/pull/51082")
# 使用 pytest 的标记 xfail 标记当前测试用例为预期失败，原因是一个特定的 GitHub pull 请求
def test_np_fix():
    # np.fix 不是一个通用函数，但其在底层由多个通用函数调用组成，同时使用了 'out' 和 'where' 关键字
    # 创建一个包含 [-1.5, -0.5, 0.5, 1.5] 的 Pandas Series 对象
    ser = pd.Series([-1.5, -0.5, 0.5, 1.5])
    # 对 Pandas Series 执行 np.fix 操作
    result = np.fix(ser)
    # 创建一个预期的 Pandas Series 对象，包含 [-1.0, -0.0, 0.0, 1.0]
    expected = pd.Series([-1.0, -0.0, 0.0, 1.0])
    # 使用 Pandas 提供的测试工具函数，比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
```