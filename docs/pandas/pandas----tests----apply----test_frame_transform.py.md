# `D:\src\scipysrc\pandas\pandas\tests\apply\test_frame_transform.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入 DataFrame、MultiIndex、Series 类
    DataFrame,
    MultiIndex,
    Series,
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块作为 tm 别名
from pandas.tests.apply.common import frame_transform_kernels  # 导入特定的测试函数
from pandas.tests.frame.common import zip_frames  # 导入用于框架比较的 zip_frames 函数


def unpack_obj(obj, klass, axis):
    """
    Helper to ensure we have the right type of object for a test parametrized
    over frame_or_series.
    """
    if klass is not DataFrame:
        obj = obj["A"]  # 如果 klass 不是 DataFrame，则选择其中的"A"列
        if axis != 0:
            pytest.skip(f"Test is only for DataFrame with axis={axis}")  # 如果 axis 不为 0，则跳过测试
    return obj  # 返回处理后的对象


def test_transform_ufunc(axis, float_frame, frame_or_series):
    # GH 35964
    obj = unpack_obj(float_frame, frame_or_series, axis)  # 获取处理后的对象

    with np.errstate(all="ignore"):
        f_sqrt = np.sqrt(obj)  # 计算 obj 的平方根

    # 使用 ufunc 进行转换
    result = obj.transform(np.sqrt, axis=axis)  # 对 obj 应用 np.sqrt 函数
    expected = f_sqrt  # 预期结果为 f_sqrt
    tm.assert_equal(result, expected)  # 使用 Pandas 测试工具比较结果


@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),  # 对单个操作（np.sqrt）进行测试，命名为 "sqrt"
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),  # 对多个操作进行测试，命名为 "absolute" 和 "sqrt"
        (np.array([np.sqrt]), ["sqrt"]),  # 使用 NumPy 数组进行单个操作测试，命名为 "sqrt"
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),  # 使用 NumPy 数组进行多个操作测试，命名为 "absolute" 和 "sqrt"
    ],
)
def test_transform_listlike(axis, float_frame, ops, names):
    # GH 35964
    other_axis = 1 if axis in {0, "index"} else 0  # 根据 axis 值确定另一个轴的索引
    with np.errstate(all="ignore"):
        expected = zip_frames([op(float_frame) for op in ops], axis=other_axis)  # 使用 zip_frames 函数对操作结果进行比较
    if axis in {0, "index"}:
        expected.columns = MultiIndex.from_product([float_frame.columns, names])  # 如果 axis 是 0 或 "index"，则设置列索引
    else:
        expected.index = MultiIndex.from_product([float_frame.index, names])  # 否则设置行索引
    result = float_frame.transform(ops, axis=axis)  # 对 float_frame 应用 ops 操作
    tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具比较结果


@pytest.mark.parametrize("ops", [[], np.array([])])
def test_transform_empty_listlike(float_frame, ops, frame_or_series):
    obj = unpack_obj(float_frame, frame_or_series, 0)  # 获取处理后的对象

    with pytest.raises(ValueError, match="No transform functions were provided"):
        obj.transform(ops)  # 对处理后的对象应用空的 ops 操作，预期引发 ValueError 异常


def test_transform_listlike_func_with_args():
    # GH 50624
    df = DataFrame({"x": [1, 2, 3]})  # 创建一个包含列 "x" 的 DataFrame

    def foo1(x, a=1, c=0):
        return x + a + c  # 定义一个带有默认参数的函数 foo1

    def foo2(x, b=2, c=0):
        return x + b + c  # 定义一个带有默认参数的函数 foo2

    msg = r"foo1\(\) got an unexpected keyword argument 'b'"
    with pytest.raises(TypeError, match=msg):
        df.transform([foo1, foo2], 0, 3, b=3, c=4)  # 测试使用带有不期望的关键字参数调用函数 foo1

    result = df.transform([foo1, foo2], 0, 3, c=4)  # 应用带有参数的函数列表对 DataFrame 进行转换
    expected = DataFrame(
        [[8, 8], [9, 9], [10, 10]],  # 预期的 DataFrame 结果
        columns=MultiIndex.from_tuples([("x", "foo1"), ("x", "foo2")]),  # 使用 MultiIndex 设置列索引
    )
    tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具比较结果


@pytest.mark.parametrize("box", [dict, Series])
def test_transform_dictlike(axis, float_frame, box):
    # GH 35964
    if axis in (0, "index"):
        e = float_frame.columns[0]  # 获取列的第一个索引
        expected = float_frame[[e]].transform(np.abs)  # 对列应用 np.abs 函数
    else:
        e = float_frame.index[0]  # 获取行的第一个索引
        expected = float_frame.iloc[[0]].transform(np.abs)  # 对行应用 np.abs 函数
    result = float_frame.transform(box({e: np.abs}), axis=axis)  # 使用 box 字典对 float_frame 进行转换
    tm.assert_frame_equal(result, expected)  # 使用 Pandas 测试工具比较结果
# 定义一个测试函数，用于测试 DataFrame 对象的 transform 方法在处理混合数据类型的字典时的行为
def test_transform_dictlike_mixed():
    # GH 40018 - mix of lists and non-lists in values of a dictionary
    # 创建一个包含列 'a', 'b', 'c' 的 DataFrame 对象
    df = DataFrame({"a": [1, 2], "b": [1, 4], "c": [1, 4]})
    # 对 DataFrame 进行 transform 操作，其中 "b" 对应的转换操作是 ["sqrt", "abs"]，"c" 对应的是 "sqrt"
    result = df.transform({"b": ["sqrt", "abs"], "c": "sqrt"})
    # 创建一个预期的 DataFrame，包含转换后的结果，列名使用 MultiIndex
    expected = DataFrame(
        [[1.0, 1, 1.0], [2.0, 4, 2.0]],
        columns=MultiIndex([("b", "c"), ("sqrt", "abs")], [(0, 0, 1), (0, 1, 0)]),
    )
    # 使用测试框架的 assert 函数比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 使用 pytest 的参数化装饰器来定义多个测试用例，测试空字典或包含不同类型操作的字典在 transform 方法中的行为
@pytest.mark.parametrize(
    "ops",
    [
        {},                           # 空字典
        {"A": []},                    # 字典中包含空列表
        {"A": [], "B": "cumsum"},     # 字典中包含一个列表和一个字符串
        {"A": "cumsum", "B": []},     # 字典中包含一个字符串和一个空列表
        {"A": [], "B": ["cumsum"]},   # 字典中包含一个空列表和一个包含字符串的列表
        {"A": ["cumsum"], "B": []},   # 字典中包含一个包含字符串的列表和一个空列表
    ],
)
# 定义测试函数，测试传入不同操作字典时，transform 方法的行为
def test_transform_empty_dictlike(float_frame, ops, frame_or_series):
    # 根据传入的参数解包 float_frame，获取待测试的对象
    obj = unpack_obj(float_frame, frame_or_series, 0)

    # 使用 pytest 的 raises 函数检查是否抛出 ValueError 异常，并验证异常消息是否包含特定文本
    with pytest.raises(ValueError, match="No transform functions were provided"):
        # 对 obj 执行 transform 方法，传入 ops 字典作为参数
        obj.transform(ops)


# 使用 pytest 的参数化装饰器定义测试用例，测试自定义函数在 transform 方法中的行为
@pytest.mark.parametrize("use_apply", [True, False])
def test_transform_udf(axis, float_frame, use_apply, frame_or_series):
    # GH 35964
    # 根据传入的参数解包 float_frame，获取待测试的对象
    obj = unpack_obj(float_frame, frame_or_series, axis)

    # 定义一个简单的自定义函数 func，根据 use_apply 参数判断是否使用 apply 方法
    def func(x):
        # 当 use_apply 为 True 且 x 不是 DataFrame 或 Series 类型时，抛出 ValueError
        if use_apply == isinstance(x, frame_or_series):
            # 强制抛出异常以回退到默认行为
            raise ValueError
        return x + 1

    # 对 obj 执行 transform 方法，传入 func 函数和 axis 参数
    result = obj.transform(func, axis=axis)
    # 计算期望结果，即 obj 中的每个元素加 1
    expected = obj + 1
    # 使用测试框架的 assert_equal 函数比较 result 和 expected 是否相等
    tm.assert_equal(result, expected)


# 定义一个不会失败的字符串列表，其中包含一些 DataFrame 的转换核心操作
wont_fail = ["ffill", "bfill", "fillna", "pad", "backfill", "shift"]
# 创建一个列表，包含 frame_transform_kernels 中的操作，但不包括 wont_fail 中的操作
frame_kernels_raise = [x for x in frame_transform_kernels if x not in wont_fail]


# 使用 pytest 的参数化装饰器定义测试用例，测试对于不支持的数据类型执行 transform 方法时的行为
@pytest.mark.parametrize("op", [*frame_kernels_raise, lambda x: x + 1])
def test_transform_bad_dtype(op, frame_or_series, request):
    # GH 35964
    # 如果操作是 "ngroup"，则使用 pytest 的 applymarker 函数标记为预期失败的测试
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    # 创建一个包含类型为 object 的列 'A' 的 DataFrame 对象，这种类型的数据大多数转换操作都会失败
    obj = DataFrame({"A": 3 * [object]})
    # 根据 frame_or_series 解包 obj，获取待测试的对象
    obj = tm.get_obj(obj, frame_or_series)
    # 定义错误类型为 TypeError
    error = TypeError
    # 定义错误消息，包含多个可能的错误提示文本，用于验证抛出的异常消息是否符合预期
    msg = "|".join(
        [
            "not supported between instances of 'type' and 'type'",
            "unsupported operand type",
        ]
    )

    # 使用 pytest 的 raises 函数检查是否抛出 TypeError 异常，并验证异常消息是否包含预期的文本
    with pytest.raises(error, match=msg):
        # 对 obj 执行 transform 方法，传入 op 作为参数
        obj.transform(op)
    with pytest.raises(error, match=msg):
        # 对 obj 执行 transform 方法，传入 [op] 作为参数
        obj.transform([op])
    with pytest.raises(error, match=msg):
        # 对 obj 执行 transform 方法，传入 {"A": op} 作为参数
        obj.transform({"A": op})
    with pytest.raises(error, match=msg):
        # 对 obj 执行 transform 方法，传入 {"A": [op]} 作为参数
        obj.transform({"A": [op]})


# 使用 pytest 的参数化装饰器定义测试用例，测试对于不支持的数据类型执行 transform 方法时的行为
@pytest.mark.parametrize("op", frame_kernels_raise)
def test_transform_failure_typeerror(request, op):
    # GH 35964

    # 如果操作是 "ngroup"，则使用 pytest 的 applymarker 函数标记为预期失败的测试
    if op == "ngroup":
        request.applymarker(
            pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
        )

    # 创建一个包含类型为 object 的列 'A' 和整数列 'B' 的 DataFrame 对象
    df = DataFrame({"A": 3 * [object], "B": [1, 2, 3]})
    # 定义错误类型为 TypeError
    error = TypeError
    # 将多个错误消息合并成一个以竖线分隔的字符串
    msg = "|".join(
        [
            "not supported between instances of 'type' and 'type'",
            "unsupported operand type",
        ]
    )
    
    # 使用 pytest 来检查是否抛出指定类型的错误，并匹配预期的错误消息
    with pytest.raises(error, match=msg):
        # 对 DataFrame 进行 transform 操作，期望抛出特定错误
        df.transform([op])
    
    with pytest.raises(error, match=msg):
        # 对 DataFrame 进行 transform 操作，期望抛出特定错误，对 A 列和 B 列分别应用 op 函数
        df.transform({"A": op, "B": op})
    
    with pytest.raises(error, match=msg):
        # 对 DataFrame 进行 transform 操作，期望抛出特定错误，对 A 列和 B 列分别应用 op 函数（作为列表传入）
        df.transform({"A": [op], "B": [op]})
    
    with pytest.raises(error, match=msg):
        # 对 DataFrame 进行 transform 操作，期望抛出特定错误，对 A 列应用 op 函数和 "shift"，对 B 列应用 op 函数
        df.transform({"A": [op, "shift"], "B": [op]})
# 定义一个测试函数，用于测试在特定条件下的数据转换失败情况（ValueError）
def test_transform_failure_valueerror():
    # 内部定义一个操作函数op，接受一个参数x
    def op(x):
        # 如果x的所有元素之和小于10，抛出ValueError异常
        if np.sum(np.sum(x)) < 10:
            raise ValueError
        return x

    # 创建一个DataFrame对象df，包含两列"A"和"B"
    df = DataFrame({"A": [1, 2, 3], "B": [400, 500, 600]})
    # 定义错误消息
    msg = "Transform function failed"

    # 使用pytest的raises断言，验证transform函数在应用op函数时是否会抛出ValueError，并匹配错误消息msg
    with pytest.raises(ValueError, match=msg):
        df.transform([op])

    # 同上，但这次使用字典形式指定列名和操作函数，验证transform函数的异常处理
    with pytest.raises(ValueError, match=msg):
        df.transform({"A": op, "B": op})

    # 类似上面，但是这次操作函数是作为列表的一部分传递，验证transform函数处理异常的情况
    with pytest.raises(ValueError, match=msg):
        df.transform({"A": [op], "B": [op]})

    # 测试多列，其中一列包含非函数元素（字符串），验证transform函数的异常处理
    with pytest.raises(ValueError, match=msg):
        df.transform({"A": [op, "shift"], "B": [op]})


# 定义一个参数化测试函数，用于测试transform函数的正常情况
@pytest.mark.parametrize("use_apply", [True, False])
def test_transform_passes_args(use_apply, frame_or_series):
    # GH 35964
    # 定义期望的位置参数和关键字参数
    expected_args = [1, 2]
    expected_kwargs = {"c": 3}

    # 定义一个函数f，接受参数x、a、b、c，用于测试transform函数的正常情况
    def f(x, a, b, c):
        # 如果use_apply与x是否为frame_or_series的实例相等，抛出ValueError异常
        if use_apply == isinstance(x, frame_or_series):
            # 强制transform函数回退
            raise ValueError
        # 断言位置参数a、b与期望的值相等
        assert [a, b] == expected_args
        # 断言关键字参数c与期望的值相等
        assert c == expected_kwargs["c"]
        return x

    # 调用frame_or_series的transform方法，应用函数f和指定的参数，用于验证transform函数的正常行为


# 定义一个测试函数，用于测试transform函数在空DataFrame上的行为
def test_transform_empty_dataframe():
    # https://github.com/pandas-dev/pandas/issues/39636
    # 创建一个空的DataFrame对象df，列名为"col1"和"col2"
    df = DataFrame([], columns=["col1", "col2"])
    # 对空DataFrame应用lambda函数，期望结果与df相等
    result = df.transform(lambda x: x + 10)
    # 使用tm.assert_frame_equal断言，验证结果与df相等
    tm.assert_frame_equal(result, df)

    # 对空DataFrame的"col1"列应用lambda函数，期望结果与df["col1"]相等
    result = df["col1"].transform(lambda x: x + 10)
    # 使用tm.assert_series_equal断言，验证结果与df["col1"]相等
    tm.assert_series_equal(result, df["col1"])
```