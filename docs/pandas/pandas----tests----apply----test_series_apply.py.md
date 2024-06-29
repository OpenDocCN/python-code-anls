# `D:\src\scipysrc\pandas\pandas\tests\apply\test_series_apply.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas as pd  # 导入 Pandas 库，用于数据处理
from pandas import (  # 从 Pandas 中导入多个模块和函数
    DataFrame,  # 数据帧对象
    Index,  # 索引对象
    MultiIndex,  # 多重索引对象
    Series,  # 序列对象
    concat,  # 连接函数
    date_range,  # 日期范围生成函数
    timedelta_range,  # 时间间隔范围生成函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块
from pandas.tests.apply.common import series_transform_kernels  # 导入 Pandas 测试模块中的特定函数

@pytest.fixture(params=[False, "compat"])  # 定义 pytest 的测试夹具，参数为 False 或 "compat"
def by_row(request):  # 定义测试夹具 by_row，根据传入参数 request 返回对应值
    return request.param

def test_series_map_box_timedelta(by_row):
    # GH#11349
    ser = Series(timedelta_range("1 day 1 s", periods=3, freq="h"))  # 创建包含时间间隔的序列对象

    def f(x):
        return x.total_seconds() if by_row else x.dt.total_seconds()  # 根据 by_row 参数选择不同的处理方式

    result = ser.apply(f, by_row=by_row)  # 应用函数 f 到序列对象 ser 上，根据 by_row 参数决定处理方式

    expected = ser.map(lambda x: x.total_seconds())  # 使用 map 方法计算序列中每个元素的总秒数
    tm.assert_series_equal(result, expected)  # 断言 result 与 expected 序列相等

    expected = Series([86401.0, 90001.0, 93601.0])  # 创建预期的结果序列
    tm.assert_series_equal(result, expected)  # 断言 result 与 expected 序列相等

def test_apply(datetime_series, by_row):
    result = datetime_series.apply(np.sqrt, by_row=by_row)  # 应用 np.sqrt 函数到日期时间序列，根据 by_row 参数决定处理方式
    with np.errstate(all="ignore"):
        expected = np.sqrt(datetime_series)  # 计算日期时间序列的每个元素的平方根
    tm.assert_series_equal(result, expected)  # 断言 result 与 expected 序列相等

    # element-wise apply (ufunc)
    result = datetime_series.apply(np.exp, by_row=by_row)  # 应用 np.exp 函数到日期时间序列，根据 by_row 参数决定处理方式
    expected = np.exp(datetime_series)  # 计算日期时间序列的每个元素的指数值
    tm.assert_series_equal(result, expected)  # 断言 result 与 expected 序列相等

    # empty series
    s = Series(dtype=object, name="foo", index=Index([], name="bar"))  # 创建空的对象序列 s
    rs = s.apply(lambda x: x, by_row=by_row)  # 应用 lambda 函数到序列 s 上，根据 by_row 参数决定处理方式
    tm.assert_series_equal(s, rs)  # 断言 s 与 rs 序列相等

    # check all metadata (GH 9322)
    assert s is not rs  # 断言 s 与 rs 不是同一个对象
    assert s.index is rs.index  # 断言 s 与 rs 的索引对象相同
    assert s.dtype == rs.dtype  # 断言 s 与 rs 的数据类型相同
    assert s.name == rs.name  # 断言 s 与 rs 的名称相同

    # index but no data
    s = Series(index=[1, 2, 3], dtype=np.float64)  # 创建带有索引但没有数据的浮点数类型序列 s
    rs = s.apply(lambda x: x, by_row=by_row)  # 应用 lambda 函数到序列 s 上，根据 by_row 参数决定处理方式
    tm.assert_series_equal(s, rs)  # 断言 s 与 rs 序列相等

def test_apply_map_same_length_inference_bug():
    s = Series([1, 2])  # 创建整数类型序列 s

    def f(x):
        return (x, x + 1)  # 定义返回元组的函数 f

    result = s.apply(f, by_row="compat")  # 应用函数 f 到序列 s 上，根据 by_row 参数决定处理方式
    expected = s.map(f)  # 使用 map 方法应用函数 f 到序列 s 上
    tm.assert_series_equal(result, expected)  # 断言 result 与 expected 序列相等

def test_apply_args():
    s = Series(["foo,bar"])  # 创建包含字符串的序列 s

    result = s.apply(str.split, args=(","))  # 应用 str.split 函数到序列 s 上，传入参数 ","
    assert result[0] == ["foo", "bar"]  # 断言 result 的第一个元素为列表 ["foo", "bar"]
    assert isinstance(result[0], list)  # 断言 result 的第一个元素是列表类型

@pytest.mark.parametrize(
    "args, kwargs, increment",
    [((), {}, 0), ((), {"a": 1}, 1), ((2, 3), {}, 32), ((1,), {"c": 2}, 201)],
)
def test_agg_args(args, kwargs, increment):
    # GH 43357
    def f(x, a=0, b=0, c=0):
        return x + a + 10 * b + 100 * c  # 定义函数 f，对参数进行加权计算

    s = Series([1, 2])  # 创建整数类型序列 s
    result = s.agg(f, 0, *args, **kwargs)  # 应用函数 f 到序列 s 上，传入额外的参数和关键字参数
    expected = s + increment  # 将序列 s 的每个元素与 increment 相加
    tm.assert_series_equal(result, expected)  # 断言 result 与 expected 序列相等

def test_agg_mapping_func_deprecated():
    # GH 53325
    s = Series([1, 2, 3])  # 创建整数类型序列 s

    def foo1(x, a=1, c=0):
        return x + a + c  # 定义函数 foo1，对参数进行加权计算

    def foo2(x, b=2, c=0):
        return x + b + c  # 定义函数 foo2，对参数进行加权计算

    s.agg(foo1, 0, 3, c=4)  # 应用函数 foo1 到序列 s 上，传入额外的参数和关键字参数
    s.agg([foo1, foo2], 0, 3, c=4)  # 应用函数列表 [foo1, foo2] 到序列 s 上，传入额外的参数和关键字参数
    s.agg({"a": foo1, "b": foo2}, 0, 3, c=4)  # 应用函数字典 {"a": foo1, "b": foo2} 到序列 s 上，传入额外的参数和关键字参数

def test_series_apply_map_box_timestamps(by_row):
    # GH#2689, GH#2627
    ser = Series(date_range("1/1/2000", periods=10))  # 创建包含日期时间的序列对象

    def func(x):
        return (x.hour, x.day, x.month)  # 定义函数 func，返回日期时间 x 的小时、天和月份信息
    # 如果不按行应用函数，则抛出 AttributeError 异常，匹配错误消息 "Series' object has no attribute 'hour'"
    if not by_row:
        msg = "Series' object has no attribute 'hour'"
        with pytest.raises(AttributeError, match=msg):
            # 在 Series 对象上应用函数 func，按行应用的方式取决于 by_row 参数
            ser.apply(func, by_row=by_row)
        # 函数结束，不再执行后续语句，直接返回
        return

    # 在 Series 对象上按行应用函数 func，根据 by_row 参数决定应用方式
    result = ser.apply(func, by_row=by_row)
    # 使用 map 方法对整个 Series 应用函数 func，生成期望的结果
    expected = ser.map(func)
    # 使用测试工具包中的 assert_series_equal 函数，断言 result 与 expected 的 Series 对象相等
    tm.assert_series_equal(result, expected)
def test_apply_box_dt64():
    # ufunc will not be boxed. Same test cases as the test_map_box

    # 创建包含两个时间戳的列表
    vals = [pd.Timestamp("2011-01-01"), pd.Timestamp("2011-01-02")]
    # 使用时间戳列表创建 Series，指定数据类型为 "M8[ns]"
    ser = Series(vals, dtype="M8[ns]")
    # 断言 Series 的数据类型为 "datetime64[ns]"
    assert ser.dtype == "datetime64[ns]"
    # 对 Series 应用 lambda 函数，生成结果为字符串格式的时间戳信息
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    # 创建预期结果的 Series
    exp = Series(["Timestamp_1_None", "Timestamp_2_None"])
    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(res, exp)


def test_apply_box_dt64tz():
    # 创建包含带时区信息的时间戳列表
    vals = [
        pd.Timestamp("2011-01-01", tz="US/Eastern"),
        pd.Timestamp("2011-01-02", tz="US/Eastern"),
    ]
    # 使用带时区信息的时间戳列表创建 Series，指定数据类型为 "M8[ns, US/Eastern]"
    ser = Series(vals, dtype="M8[ns, US/Eastern]")
    # 断言 Series 的数据类型为 "datetime64[ns, US/Eastern]"
    assert ser.dtype == "datetime64[ns, US/Eastern]"
    # 对 Series 应用 lambda 函数，生成带时间戳类型和时区的字符串信息
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.day}_{x.tz}", by_row="compat")
    # 创建预期结果的 Series
    exp = Series(["Timestamp_1_US/Eastern", "Timestamp_2_US/Eastern"])
    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(res, exp)


def test_apply_box_td64():
    # timedelta 类型
    vals = [pd.Timedelta("1 days"), pd.Timedelta("2 days")]
    # 使用 timedelta 列表创建 Series
    ser = Series(vals)
    # 断言 Series 的数据类型为 "timedelta64[ns]"
    assert ser.dtype == "timedelta64[ns]"
    # 对 Series 应用 lambda 函数，生成带 timedelta 类型的字符串信息
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.days}", by_row="compat")
    # 创建预期结果的 Series
    exp = Series(["Timedelta_1", "Timedelta_2"])
    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(res, exp)


def test_apply_box_period():
    # period 类型
    vals = [pd.Period("2011-01-01", freq="M"), pd.Period("2011-01-02", freq="M")]
    # 使用 period 列表创建 Series
    ser = Series(vals)
    # 断言 Series 的数据类型为 "Period[M]"
    assert ser.dtype == "Period[M]"
    # 对 Series 应用 lambda 函数，生成带 period 类型和频率字符串的信息
    res = ser.apply(lambda x: f"{type(x).__name__}_{x.freqstr}", by_row="compat")
    # 创建预期结果的 Series
    exp = Series(["Period_M", "Period_M"])
    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(res, exp)


def test_apply_datetimetz(by_row):
    # 创建带时区的日期时间序列
    values = date_range("2011-01-01", "2011-01-02", freq="h").tz_localize("Asia/Tokyo")
    s = Series(values, name="XX")

    # 对序列应用 lambda 函数，对日期时间进行偏移，结果仍保留时区信息
    result = s.apply(lambda x: x + pd.offsets.Day(), by_row=by_row)
    # 创建预期结果的 Series
    exp_values = date_range("2011-01-02", "2011-01-03", freq="h").tz_localize(
        "Asia/Tokyo"
    )
    exp = Series(exp_values, name="XX")
    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(result, exp)

    # 对序列应用 lambda 函数，获取小时数，根据 by_row 参数确定返回结果的数据类型
    result = s.apply(lambda x: x.hour if by_row else x.dt.hour, by_row=by_row)
    # 创建预期结果的 Series
    exp = Series(list(range(24)) + [0], name="XX", dtype="int64" if by_row else "int32")
    # 断言结果 Series 和预期 Series 相等
    tm.assert_series_equal(result, exp)

    # 非向量化操作
    def f(x):
        return str(x.tz) if by_row else str(x.dt.tz)

    # 对序列应用函数 f，根据 by_row 参数确定返回结果的数据类型
    result = s.apply(f, by_row=by_row)
    if by_row:
        exp = Series(["Asia/Tokyo"] * 25, name="XX")
        # 断言结果 Series 和预期 Series 相等
        tm.assert_series_equal(result, exp)
    else:
        assert result == "Asia/Tokyo"


def test_apply_categorical(by_row, using_infer_string):
    # 创建有序分类数据
    values = pd.Categorical(list("ABBABCD"), categories=list("DCBA"), ordered=True)
    ser = Series(values, name="XX", index=list("abcdefg"))

    if not by_row:
        # 如果不是按行处理，则测试是否抛出预期的 AttributeError 异常
        msg = "Series' object has no attribute 'lower"
        with pytest.raises(AttributeError, match=msg):
            ser.apply(lambda x: x.lower(), by_row=by_row)
        # 断言应用 lambda 函数返回值为 "A"
        assert ser.apply(lambda x: "A", by_row=by_row) == "A"
        return
    # 使用 apply 方法对序列 ser 中的每个元素执行小写转换操作
    result = ser.apply(lambda x: x.lower(), by_row=by_row)

    # 创建一个有序的分类变量 values，基于指定的分类和顺序
    values = pd.Categorical(list("abbabcd"), categories=list("dcba"), ordered=True)
    # 创建预期的 Series 对象 exp，使用上述分类变量 values，指定名称和索引
    exp = Series(values, name="XX", index=list("abcdefg"))
    # 断言 result 和 exp 应该相等，即 Series 的内容和结构应该一致
    tm.assert_series_equal(result, exp)
    # 断言 result.values 和 exp.values 应该是相等的分类变量
    tm.assert_categorical_equal(result.values, exp.values)

    # 使用 apply 方法对序列 ser 中的每个元素应用固定的返回值 "A"
    result = ser.apply(lambda x: "A")
    # 创建预期的 Series 对象 exp，包含 7 个 "A"，指定名称和索引
    exp = Series(["A"] * 7, name="XX", index=list("abcdefg"))
    # 断言 result 和 exp 应该相等，即 Series 的内容和结构应该一致
    assert result.dtype == object if not using_infer_string else "string[pyarrow_numpy]"
# 使用 pytest 的参数化装饰器来定义多个测试用例，每个 series 是一个测试用例的输入
@pytest.mark.parametrize("series", [["1-1", "1-1", np.nan], ["1-1", "1-2", np.nan]])
def test_apply_categorical_with_nan_values(series, by_row):
    # 根据 GitHub 上的 issue 编号指出，此处修复了 GH 20714，并在 GH 24275 中进行了确认
    s = Series(series, dtype="category")
    
    # 如果不按行处理
    if not by_row:
        msg = "'Series' object has no attribute 'split'"
        # 使用 pytest 来检查是否抛出预期的 AttributeError 异常，并匹配特定的错误信息
        with pytest.raises(AttributeError, match=msg):
            s.apply(lambda x: x.split("-")[0], by_row=by_row)
        return

    # 按行处理，应用 lambda 函数来对每个元素进行处理，提取第一个部分，并转换为 object 类型
    result = s.apply(lambda x: x.split("-")[0], by_row=by_row)
    result = result.astype(object)
    
    # 期望的结果，也是一个 Series 对象，转换为 object 类型
    expected = Series(["1", "1", np.nan], dtype="category")
    expected = expected.astype(object)
    
    # 使用 pytest 的 assert_series_equal 来比较实际结果和期望结果
    tm.assert_series_equal(result, expected)


def test_apply_empty_integer_series_with_datetime_index(by_row):
    # 根据 GitHub 上的 issue 编号 GH 21245
    # 创建一个空的整数 Series，具有日期时间索引
    s = Series([], index=date_range(start="2018-01-01", periods=0), dtype=int)
    
    # 应用 lambda 函数来对每个元素进行处理，这里不会改变结果，只是复制返回
    result = s.apply(lambda x: x, by_row=by_row)
    
    # 使用 pytest 的 assert_series_equal 来比较实际结果和期望结果
    tm.assert_series_equal(result, s)


def test_apply_dataframe_iloc():
    # 创建一个包含 np.uint64 的 DataFrame 和另一个 DataFrame 用于索引
    uintDF = DataFrame(np.uint64([1, 2, 3, 4, 5]), columns=["Numbers"])
    indexDF = DataFrame([2, 3, 2, 1, 2], columns=["Indices"])

    # 定义一个函数来提取目标行中的值
    def retrieve(targetRow, targetDF):
        val = targetDF["Numbers"].iloc[targetRow]
        return val

    # 使用 apply 函数将 retrieve 应用于 indexDF 的 "Indices" 列
    result = indexDF["Indices"].apply(retrieve, args=(uintDF,))
    
    # 期望的结果，是一个 Series 对象，dtype 为 uint64
    expected = Series([3, 4, 3, 2, 3], name="Indices", dtype="uint64")
    
    # 使用 pytest 的 assert_series_equal 来比较实际结果和期望结果
    tm.assert_series_equal(result, expected)


def test_transform(string_series, by_row):
    # 使用 np.errstate 来忽略所有的异常
    with np.errstate(all="ignore"):
        # 对 string_series 应用 np.sqrt 函数，计算平方根并复制结果
        f_sqrt = np.sqrt(string_series)
        # 对 string_series 应用 np.abs 函数，计算绝对值并复制结果
        f_abs = np.abs(string_series)

        # 使用 apply 函数将 np.sqrt 应用于 string_series，by_row 参数表示按行处理
        result = string_series.apply(np.sqrt, by_row=by_row)
        expected = f_sqrt.copy()
        
        # 使用 pytest 的 assert_series_equal 来比较实际结果和期望结果
        tm.assert_series_equal(result, expected)

        # 使用 apply 函数将 [np.sqrt] 应用于 string_series，by_row 参数表示按行处理
        result = string_series.apply([np.sqrt], by_row=by_row)
        expected = f_sqrt.to_frame().copy()
        expected.columns = ["sqrt"]
        
        # 使用 pytest 的 assert_frame_equal 来比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

        # 使用 apply 函数将 ["sqrt"] 应用于 string_series，by_row 参数表示按行处理
        result = string_series.apply(["sqrt"], by_row=by_row)
        
        # 使用 pytest 的 assert_frame_equal 来比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

        # 使用 apply 函数将 [np.sqrt, np.abs] 应用于 string_series，by_row 参数表示按行处理
        result = string_series.apply([np.sqrt, np.abs], by_row=by_row)
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["sqrt", "absolute"]
        
        # 使用 pytest 的 assert_frame_equal 来比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

        # 使用 apply 函数将 {"foo": np.sqrt, "bar": np.abs} 应用于 string_series，by_row 参数表示按行处理
        result = string_series.apply({"foo": np.sqrt, "bar": np.abs}, by_row=by_row)
        expected = concat([f_sqrt, f_abs], axis=1)
        expected.columns = ["foo", "bar"]
        expected = expected.unstack().rename("series")
        
        # 使用 pytest 的 assert_series_equal 来比较实际结果和期望结果，通过 reindex_like 对齐索引
        tm.assert_series_equal(result.reindex_like(expected), expected)


@pytest.mark.parametrize("op", series_transform_kernels)
def test_transform_partial_failure(op, request):
    # 根据 GitHub 上的 issue 编号 GH 35964
    # 此处测试部分失败的情况
    # 如果操作符是 "ffill", "bfill" 或者 "shift" 中的一个，添加一个标记来表示预期该操作会失败
    if op in ("ffill", "bfill", "shift"):
        request.applymarker(
            pytest.mark.xfail(reason=f"{op} is successful on any dtype")
        )

    # 使用 object 类型会导致大多数转换内核失败
    ser = Series(3 * [object])

    # 如果操作符是 "fillna" 或者 "ngroup"，则定义错误类型为 ValueError，并设置错误消息
    if op in ("fillna", "ngroup"):
        error = ValueError
        msg = "Transform function failed"
    else:
        # 否则，定义错误类型为 TypeError，并设置详细的错误消息
        error = TypeError
        msg = "|".join(
            [
                "not supported between instances of 'type' and 'type'",
                "unsupported operand type",
            ]
        )

    # 使用 pytest 来检测是否会抛出特定类型的错误，并匹配相应的错误消息
    with pytest.raises(error, match=msg):
        ser.transform([op, "shift"])

    with pytest.raises(error, match=msg):
        ser.transform({"A": op, "B": "shift"})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op], "B": ["shift"]})

    with pytest.raises(error, match=msg):
        ser.transform({"A": [op, "shift"], "B": [op]})
# GH 40211
def test_transform_partial_failure_valueerror():
    # 定义一个空操作函数，返回其输入参数
    def noop(x):
        return x

    # 定义一个抛出 ValueError 异常的操作函数
    def raising_op(_):
        raise ValueError

    # 创建一个包含三个 object 类型元素的 Series 对象
    ser = Series(3 * [object])
    # 设置错误信息提示字符串
    msg = "Transform function failed"

    # 测试使用 transform 方法，期望捕获 ValueError 异常，并匹配指定的错误信息
    with pytest.raises(ValueError, match=msg):
        ser.transform([noop, raising_op])

    # 测试使用 transform 方法，期望捕获 ValueError 异常，并匹配指定的错误信息
    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": raising_op, "B": noop})

    # 测试使用 transform 方法，期望捕获 ValueError 异常，并匹配指定的错误信息
    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [raising_op], "B": [noop]})

    # 测试使用 transform 方法，期望捕获 ValueError 异常，并匹配指定的错误信息
    with pytest.raises(ValueError, match=msg):
        ser.transform({"A": [noop, raising_op], "B": [noop]})


# demonstration tests
def test_demo():
    # 创建一个包含整数的 Series 对象，范围从 0 到 5
    s = Series(range(6), dtype="int64", name="series")

    # 测试 agg 方法，计算最小值和最大值
    result = s.agg(["min", "max"])
    expected = Series([0, 5], index=["min", "max"], name="series")
    tm.assert_series_equal(result, expected)

    # 测试 agg 方法，使用字典参数指定计算最小值
    result = s.agg({"foo": "min"})
    expected = Series([0], index=["foo"], name="series")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("func", [str, lambda x: str(x)])
def test_apply_map_evaluate_lambdas_the_same(string_series, func, by_row):
    # 测试 apply 方法和 map 方法在 by_row="compat" 时的行为
    # 和其他情况下的向量化计算行为
    result = string_series.apply(func, by_row=by_row)

    if by_row:
        # 如果 by_row 为 True，则期望结果与 map(func) 的结果相同
        expected = string_series.map(func)
        tm.assert_series_equal(result, expected)
    else:
        # 否则，期望结果为 string_series 的字符串表示
        assert result == str(string_series)


def test_agg_evaluate_lambdas(string_series):
    # GH53325
    # 测试 agg 方法对于 lambda 函数的评估行为
    result = string_series.agg(lambda x: type(x))
    assert result is Series

    # 测试 agg 方法对于函数类型的评估行为
    result = string_series.agg(type)
    assert result is Series


@pytest.mark.parametrize("op_name", ["agg", "apply"])
def test_with_nested_series(datetime_series, op_name):
    # GH 2316 & GH52123
    # 测试 .agg 方法在使用 reducer 和 transform 时的行为
    result = getattr(datetime_series, op_name)(
        lambda x: Series([x, x**2], index=["x", "x^2"])
    )
    if op_name == "apply":
        # 如果 op_name 为 "apply"，期望结果为 DataFrame 对象
        expected = DataFrame({"x": datetime_series, "x^2": datetime_series**2})
        tm.assert_frame_equal(result, expected)
    else:
        # 否则，期望结果为包含两个元素的 Series 对象
        expected = Series([datetime_series, datetime_series**2], index=["x", "x^2"])
        tm.assert_series_equal(result, expected)


def test_replicate_describe(string_series):
    # this also tests a result set that is all scalars
    # 测试 apply 方法使用包含描述统计方法的字典参数的行为
    expected = string_series.describe()
    result = string_series.apply(
        {
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "25%": lambda x: x.quantile(0.25),
            "50%": "median",
            "75%": lambda x: x.quantile(0.75),
            "max": "max",
        },
    )
    tm.assert_series_equal(result, expected)


def test_reduce(string_series):
    # reductions with named functions
    # 测试 agg 方法使用命名函数进行聚合操作
    result = string_series.agg(["sum", "mean"])
    # 创建一个期望的 Series 对象，包含两个统计量：总和和平均值
    expected = Series(
        [string_series.sum(), string_series.mean()],  # 使用 string_series 的 sum() 和 mean() 方法计算总和和平均值
        ["sum", "mean"],  # 设置 Series 的索引标签为 "sum" 和 "mean"
        name=string_series.name,  # 设置 Series 的名称为 string_series 的名称
    )
    
    # 使用测试工具包中的方法来比较 result 和 expected 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "how, kwds",
    [("agg", {}), ("apply", {"by_row": "compat"}), ("apply", {"by_row": False})],
)
# 定义参数化测试，参数包括操作方式和关键字参数
def test_non_callable_aggregates(how, kwds):
    # 测试对非可调用序列属性使用agg函数
    # GH 39116 - 将测试扩展到apply函数
    s = Series([1, 2, None])

    # 使用getattr调用agg方法，相当于调用s.size
    result = getattr(s, how)("size", **kwds)
    expected = s.size
    assert result == expected

    # 测试与可调用的聚合函数混合使用的情况
    result = getattr(s, how)(["size", "count", "mean"], **kwds)
    expected = Series({"size": 3.0, "count": 2.0, "mean": 1.5})
    tm.assert_series_equal(result, expected)

    result = getattr(s, how)({"size": "size", "count": "count", "mean": "mean"}, **kwds)
    tm.assert_series_equal(result, expected)


def test_series_apply_no_suffix_index(by_row):
    # GH36189
    # 测试在不添加后缀索引的情况下应用apply函数
    s = Series([4] * 3)
    result = s.apply(["sum", lambda x: x.sum(), lambda x: x.sum()], by_row=by_row)
    expected = Series([12, 12, 12], index=["sum", "<lambda>", "<lambda>"])

    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "dti,exp",
    [
        (
            Series([1, 2], index=pd.DatetimeIndex([0, 31536000000])),
            DataFrame(np.repeat([[1, 2]], 2, axis=0), dtype="int64"),
        ),
        (
            Series(
                np.arange(10, dtype=np.float64),
                index=date_range("2020-01-01", periods=10),
                name="ts",
            ),
            DataFrame(np.repeat([[1, 2]], 10, axis=0), dtype="int64"),
        ),
    ],
)
@pytest.mark.parametrize("aware", [True, False])
def test_apply_series_on_date_time_index_aware_series(dti, exp, aware):
    # GH 25959
    # 在本地化时间序列上调用apply不应导致错误
    if aware:
        index = dti.tz_localize("UTC").index
    else:
        index = dti.index
    result = Series(index).apply(lambda x: Series([1, 2]))
    tm.assert_frame_equal(result, exp)


@pytest.mark.parametrize(
    "by_row, expected", [("compat", Series(np.ones(10), dtype="int64")), (False, 1)]
)
def test_apply_scalar_on_date_time_index_aware_series(by_row, expected):
    # GH 25959
    # 在本地化时间序列上调用apply不应导致错误
    series = Series(
        np.arange(10, dtype=np.float64),
        index=date_range("2020-01-01", periods=10, tz="UTC"),
    )
    result = Series(series.index).apply(lambda x: 1, by_row=by_row)
    tm.assert_equal(result, expected)


def test_apply_to_timedelta(by_row):
    list_of_valid_strings = ["00:00:01", "00:00:02"]
    a = pd.to_timedelta(list_of_valid_strings)
    b = Series(list_of_valid_strings).apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)

    list_of_strings = ["00:00:01", np.nan, pd.NaT, pd.NaT]

    a = pd.to_timedelta(list_of_strings)
    ser = Series(list_of_strings)
    b = ser.apply(pd.to_timedelta, by_row=by_row)
    tm.assert_series_equal(Series(a), b)
    "ops, names",
    # 创建一个包含多个元组的列表，每个元组包含一组函数和对应的函数名列表
    [
        ([np.sum], ["sum"]),  # 第一个元组包含单个函数 np.sum 和对应的函数名列表 ["sum"]
        ([np.sum, np.mean], ["sum", "mean"]),  # 第二个元组包含两个函数 np.sum 和 np.mean，以及对应的函数名列表 ["sum", "mean"]
        (np.array([np.sum]), ["sum"]),  # 第三个元组包含一个数组，数组中包含单个函数 np.sum，以及对应的函数名列表 ["sum"]
        (np.array([np.sum, np.mean]), ["sum", "mean"]),  # 第四个元组包含一个数组，数组中包含两个函数 np.sum 和 np.mean，以及对应的函数名列表 ["sum", "mean"]
    ],
# 在这个模块中引入必要的测试框架 pytest
import pytest

# 参数化装饰器，用于给函数 test_apply_listlike_reducer 多次传入不同的参数进行测试
@pytest.mark.parametrize(
    "how, kwargs",
    [["agg", {}], ["apply", {"by_row": "compat"}], ["apply", {"by_row": False}]],
)
# 测试函数，用于测试在给定的 string_series 上应用 ops 操作后的结果
def test_apply_listlike_reducer(string_series, ops, names, how, kwargs):
    # GH 39140
    # 创建一个期望结果的 Series 对象，对 string_series 应用 ops 中的操作并赋予名字 "series"
    expected = Series({name: op(string_series) for name, op in zip(names, ops)})
    expected.name = "series"
    # 在 string_series 上调用指定的方法（由 how 参数指定），并传入其他关键字参数 kwargs
    result = getattr(string_series, how)(ops, **kwargs)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 参数化装饰器，用于给函数 test_apply_dictlike_reducer 多次传入不同的参数进行测试
@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sum},
        {"A": np.sum, "B": np.mean},
        Series({"A": np.sum}),
        Series({"A": np.sum, "B": np.mean}),
    ],
)
# 参数化装饰器，用于给函数 test_apply_dictlike_reducer 多次传入不同的参数进行测试
@pytest.mark.parametrize(
    "how, kwargs",
    [["agg", {}], ["apply", {"by_row": "compat"}], ["apply", {"by_row": False}]],
)
# 测试函数，用于测试在给定的 string_series 上应用 ops 操作后的结果
def test_apply_dictlike_reducer(string_series, ops, how, kwargs, by_row):
    # GH 39140
    # 创建一个期望结果的 Series 对象，对 string_series 应用 ops 中的操作
    expected = Series({name: op(string_series) for name, op in ops.items()})
    expected.name = string_series.name
    # 在 string_series 上调用指定的方法（由 how 参数指定），并传入其他关键字参数 kwargs
    result = getattr(string_series, how)(ops, **kwargs)
    # 断言两个 Series 对象是否相等
    tm.assert_series_equal(result, expected)


# 参数化装饰器，用于给函数 test_apply_listlike_transformer 多次传入不同的参数进行测试
@pytest.mark.parametrize(
    "ops, names",
    [
        ([np.sqrt], ["sqrt"]),
        ([np.abs, np.sqrt], ["absolute", "sqrt"]),
        (np.array([np.sqrt]), ["sqrt"]),
        (np.array([np.abs, np.sqrt]), ["absolute", "sqrt"]),
    ],
)
# 测试函数，用于测试在给定的 string_series 上应用 ops 中的转换操作后的结果
def test_apply_listlike_transformer(string_series, ops, names, by_row):
    # GH 39140
    # 忽略 numpy 中的错误
    with np.errstate(all="ignore"):
        # 创建一个期望结果的 DataFrame 对象，对 string_series 应用 ops 中的操作
        expected = concat([op(string_series) for op in ops], axis=1)
        expected.columns = names
        # 在 string_series 上调用 apply 方法应用 ops 中的操作，并传入 by_row 参数
        result = string_series.apply(ops, by_row=by_row)
        # 断言两个 DataFrame 对象是否相等
        tm.assert_frame_equal(result, expected)


# 参数化装饰器，用于给函数 test_apply_listlike_lambda 多次传入不同的参数进行测试
@pytest.mark.parametrize(
    "ops, expected",
    [
        ([lambda x: x], DataFrame({"<lambda>": [1, 2, 3]})),
        ([lambda x: x.sum()], Series([6], index=["<lambda>"])),
    ],
)
# 测试函数，用于测试在给定的 Series 对象上应用 ops 中的 lambda 函数后的结果
def test_apply_listlike_lambda(ops, expected, by_row):
    # GH53400
    # 创建一个 Series 对象
    ser = Series([1, 2, 3])
    # 在 ser 上调用 apply 方法应用 ops 中的 lambda 函数
    result = ser.apply(ops, by_row=by_row)
    # 断言两个结果对象是否相等
    tm.assert_equal(result, expected)


# 参数化装饰器，用于给函数 test_apply_dictlike_transformer 多次传入不同的参数进行测试
@pytest.mark.parametrize(
    "ops",
    [
        {"A": np.sqrt},
        {"A": np.sqrt, "B": np.exp},
        Series({"A": np.sqrt}),
        Series({"A": np.sqrt, "B": np.exp}),
    ],
)
# 测试函数，用于测试在给定的 Series 对象上应用 ops 中的转换操作后的结果
def test_apply_dictlike_transformer(string_series, ops, by_row):
    # GH 39140
    # 忽略 numpy 中的错误
    with np.errstate(all="ignore"):
        # 创建一个期望结果的 Series 对象，对 string_series 应用 ops 中的操作
        expected = concat({name: op(string_series) for name, op in ops.items()})
        expected.name = string_series.name
        # 在 string_series 上调用 apply 方法应用 ops 中的操作，并传入 by_row 参数
        result = string_series.apply(ops, by_row=by_row)
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)


# 参数化装饰器，用于给函数 test_apply_dictlike_lambda 多次传入不同的参数进行测试
@pytest.mark.parametrize(
    "ops, expected",
    [
        (
            {"a": lambda x: x},
            Series([1, 2, 3], index=MultiIndex.from_arrays([["a"] * 3, range(3)])),
        ),
        ({"a": lambda x: x.sum()}, Series([6], index=["a"])),
    ],
)
# 测试函数，用于测试在给定的 Series 对象上应用 ops 中的 lambda 函数后的结果
def test_apply_dictlike_lambda(ops, by_row, expected):
    # GH53400
    # 创建一个 Series 对象
    ser = Series([1, 2, 3])
    # 在 ser 上调用 apply 方法应用 ops 中的 lambda 函数
    result = ser.apply(ops, by_row=by_row)
    # 断言两个结果对象是否相等
    tm.assert_equal(result, expected)
# 定义一个测试函数，用于验证 apply 方法在保留列名时的行为
def test_apply_retains_column_name(by_row):
    # GH 16380
    # 创建一个包含单列 'x' 的 DataFrame，其索引为整数范围 0 到 2，并命名为 'x'
    df = DataFrame({"x": range(3)}, Index(range(3), name="x"))
    # 对列 'x' 应用一个 lambda 函数，该函数返回基于 x 值的 Series，索引也是整数范围 0 到 x，并命名为 'y'
    result = df.x.apply(lambda x: Series(range(x + 1), Index(range(x + 1), name="y")))
    # 创建预期的 DataFrame，包含由列表组成的值，列索引为整数范围 0 到 2，行索引为整数范围 0 到 2，分别命名为 'y' 和 'x'
    expected = DataFrame(
        [[0.0, np.nan, np.nan], [0.0, 1.0, np.nan], [0.0, 1.0, 2.0]],
        columns=Index(range(3), name="y"),
        index=Index(range(3), name="x"),
    )
    # 使用测试框架中的方法验证结果 DataFrame 是否与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于验证 apply 方法在处理类型函数时的行为
def test_apply_type():
    # GH 46719
    # 创建一个包含不同类型的 Series，索引为 ['a', 'b', 'c']，值分别为整数 3、字符串 "string" 和类型 float
    s = Series([3, "string", float], index=["a", "b", "c"])
    # 对 Series 应用 type 函数，返回一个 Series，其值为对应元素的类型
    result = s.apply(type)
    # 创建预期的 Series，包含对应元素的类型，索引与输入 Series 相同
    expected = Series([int, str, type], index=["a", "b", "c"])
    # 使用测试框架中的方法验证结果 Series 是否与预期的 Series 相等
    tm.assert_series_equal(result, expected)


# 定义一个测试函数，用于验证 apply 方法在解包嵌套数据时的行为
def test_series_apply_unpack_nested_data():
    # GH#55189
    # 创建一个包含两个列表的 Series
    ser = Series([[1, 2, 3], [4, 5, 6, 7]])
    # 对 Series 应用 lambda 函数，将每个列表转换为 Series，得到一个 DataFrame
    result = ser.apply(lambda x: Series(x))
    # 创建预期的 DataFrame，包含从列表转换而来的数据，其中列名为整数 0 到 3，行索引为整数 0 到 1
    expected = DataFrame({0: [1.0, 4.0], 1: [2.0, 5.0], 2: [3.0, 6.0], 3: [np.nan, 7]})
    # 使用测试框架中的方法验证结果 DataFrame 是否与预期的 DataFrame 相等
    tm.assert_frame_equal(result, expected)
```