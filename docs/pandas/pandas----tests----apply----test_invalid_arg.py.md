# `D:\src\scipysrc\pandas\pandas\tests\apply\test_invalid_arg.py`

```
# 导入需要的模块和库
from itertools import chain  # 导入chain函数，用于迭代器操作
import re  # 导入re模块，用于正则表达式操作

import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas.errors import SpecificationError  # 从pandas.errors模块导入SpecificationError异常类

from pandas import (  # 从pandas库导入以下模块：
    DataFrame,  # DataFrame数据结构，用于操作二维标记数据
    Series,  # Series数据结构，用于操作一维标记数据
    date_range,  # 生成日期范围的函数
)
import pandas._testing as tm  # 导入pandas._testing模块，用于测试工具

# 测试函数：测试不合法的result_type参数
@pytest.mark.parametrize("result_type", ["foo", 1])
def test_result_type_error(result_type):
    # 创建一个DataFrame对象，包含重复的行向量加一的数据
    df = DataFrame(
        np.tile(np.arange(3, dtype="int64"), 6).reshape(6, -1) + 1,
        columns=["A", "B", "C"],
    )

    # 定义错误信息
    msg = (
        "invalid value for result_type, must be one of "
        "{None, 'reduce', 'broadcast', 'expand'}"
    )
    # 断言应该抛出ValueError异常，并匹配msg消息
    with pytest.raises(ValueError, match=msg):
        df.apply(lambda x: [1, 2, 3], axis=1, result_type=result_type)


# 测试函数：测试不合法的axis值
def test_apply_invalid_axis_value():
    # 创建一个DataFrame对象，包含列表的列表数据
    df = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=["a", "a", "c"])
    # 定义错误信息
    msg = "No axis named 2 for object type DataFrame"
    # 断言应该抛出ValueError异常，并匹配msg消息
    with pytest.raises(ValueError, match=msg):
        df.apply(lambda x: x, 2)


# 测试函数：测试agg方法未提供参数时的异常
def test_agg_raises():
    # 创建一个DataFrame对象，包含字典数据
    df = DataFrame({"A": [0, 1], "B": [1, 2]})
    # 定义错误信息
    msg = "Must provide"
    # 断言应该抛出TypeError异常，并匹配msg消息
    with pytest.raises(TypeError, match=msg):
        df.agg()


# 测试函数：测试map方法不合法的na_action参数
def test_map_with_invalid_na_action_raises():
    # 创建一个Series对象，包含整数列表数据
    s = Series([1, 2, 3])
    # 定义错误信息
    msg = "na_action must either be 'ignore' or None"
    # 断言应该抛出ValueError异常，并匹配msg消息
    with pytest.raises(ValueError, match=msg):
        s.map(lambda x: x, na_action="____")


# 测试函数：测试map方法参数为字典时不合法的na_action参数
@pytest.mark.parametrize("input_na_action", ["____", True])
def test_map_arg_is_dict_with_invalid_na_action_raises(input_na_action):
    # 创建一个Series对象，包含整数列表数据
    s = Series([1, 2, 3])
    # 定义错误信息
    msg = f"na_action must either be 'ignore' or None, {input_na_action} was passed"
    # 断言应该抛出ValueError异常，并匹配msg消息
    with pytest.raises(ValueError, match=msg):
        s.map({1: 2}, na_action=input_na_action)


# 测试函数：测试apply、agg和transform方法中的嵌套重命名不支持异常
@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
@pytest.mark.parametrize("func", [{"A": {"B": "sum"}}, {"A": {"B": ["sum"]}}])
def test_nested_renamer(frame_or_series, method, func):
    # 创建一个DataFrame或Series对象，根据frame_or_series函数的返回值确定
    obj = frame_or_series({"A": [1]})
    # 定义错误信息
    match = "nested renamer is not supported"
    # 断言应该抛出SpecificationError异常，并匹配match消息
    with pytest.raises(SpecificationError, match=match):
        getattr(obj, method)(func)


# 测试函数：测试Series对象中嵌套重命名不支持异常
@pytest.mark.parametrize(
    "renamer",
    [{"foo": ["min", "max"]}, {"foo": ["min", "max"], "bar": ["sum", "mean"]}],
)
def test_series_nested_renamer(renamer):
    # 创建一个Series对象，包含整数范围数据
    s = Series(range(6), dtype="int64", name="series")
    # 定义错误信息
    msg = "nested renamer is not supported"
    # 断言应该抛出SpecificationError异常，并匹配msg消息
    with pytest.raises(SpecificationError, match=msg):
        s.agg(renamer)
    # 使用随机数生成器创建一个 DataFrame，包含 10 行和 3 列，列名分别为 "A", "B", "C"，行索引从 "1/1/2000" 开始，持续 10 个时间周期
    tsdf = DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        columns=["A", "B", "C"],
        index=date_range("1/1/2000", periods=10),
    )
    
    # 定义错误消息字符串
    msg = "nested renamer is not supported"
    
    # 使用 pytest 中的 `raises` 方法验证是否会抛出 SpecificationError 异常，并检查异常消息是否与预期的错误消息匹配
    with pytest.raises(SpecificationError, match=msg):
        # 对 tsdf.A 应用聚合操作，指定了一个无法支持的嵌套的重命名器
        tsdf.A.agg({"foo": ["sum", "mean"]})
# 使用 pytest 的参数化装饰器，依次测试方法为 "agg" 和 "transform"
@pytest.mark.parametrize("method", ["agg", "transform"])
def test_dict_nested_renaming_depr(method):
    # 创建一个 DataFrame 包含列 "A" 和 "B"
    df = DataFrame({"A": range(5), "B": 5})

    # 使用 pytest 的断言，检查在指定操作方法中执行嵌套重命名是否会引发 SpecificationError 异常
    msg = r"nested renamer is not supported"
    with pytest.raises(SpecificationError, match=msg):
        getattr(df, method)({"A": {"foo": "min"}, "B": {"bar": "max"}})


# 使用 pytest 的参数化装饰器，依次测试方法为 "apply", "agg", "transform" 和 func 参数为 {"B": "sum"} 或 {"B": ["sum"]}
@pytest.mark.parametrize("method", ["apply", "agg", "transform"])
@pytest.mark.parametrize("func", [{"B": "sum"}, {"B": ["sum"]}])
def test_missing_column(method, func):
    # 创建一个 DataFrame 包含列 "A"，在特定操作中测试缺少列 "B" 是否引发 KeyError 异常
    # GH 40004
    obj = DataFrame({"A": [1]})
    msg = r"Label\(s\) \['B'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        getattr(obj, method)(func)


# 测试 DataFrame 在 transform 操作中使用不同列名数据类型时是否引发 KeyError 异常
def test_transform_mixed_column_name_dtypes():
    # GH39025
    df = DataFrame({"a": ["1"]})
    msg = r"Label\(s\) \[1, 'b'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.transform({"a": int, 1: str, "b": int})


# 使用 pytest 的参数化装饰器，依次测试方法 "pct_change", "nsmallest", "tail" 和其对应的参数
@pytest.mark.parametrize(
    "how, args", [("pct_change", ()), ("nsmallest", (1, ["a", "b"])), ("tail", 1)]
)
def test_apply_str_axis_1_raises(how, args):
    # 创建一个 DataFrame 包含列 "a" 和 "b"，测试特定操作如何不支持 axis=1 是否引发 ValueError 异常
    # GH 39211 - some ops don't support axis=1
    df = DataFrame({"a": [1, 2], "b": [3, 4]})
    msg = f"Operation {how} does not support axis=1"
    with pytest.raises(ValueError, match=msg):
        df.apply(how, axis=1, args=args)


# 测试 Series 在 transform 操作中使用 axis=1 是否引发 ValueError 异常
def test_transform_axis_1_raises():
    # GH 35964
    msg = "No axis named 1 for object type Series"
    with pytest.raises(ValueError, match=msg):
        Series([1]).transform("sum", axis=1)


# 测试 apply 方法在自定义函数中，当发生 AttributeError 时是否会抛出异常
def test_apply_modify_traceback():
    # 创建包含多列的 DataFrame，修改一个值为 NaN，然后测试 apply 方法在自定义函数中是否会抛出 AttributeError 异常
    data = DataFrame(
        {
            "A": [
                "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"
            ],
            "B": [
                "one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"
            ],
            "C": [
                "dull", "dull", "shiny", "dull", "dull", "shiny", "shiny", "dull", "shiny", "shiny", "shiny"
            ],
            "D": np.random.default_rng(2).standard_normal(11),
            "E": np.random.default_rng(2).standard_normal(11),
            "F": np.random.default_rng(2).standard_normal(11),
        }
    )

    # 将某个位置的值设置为 NaN
    data.loc[4, "C"] = np.nan

    # 定义一个自定义函数，测试其在 apply 方法中是否会抛出 AttributeError 异常
    def transform(row):
        if row["C"].startswith("shin") and row["A"] == "foo":
            row["D"] = 7
        return row

    msg = "'float' object has no attribute 'startswith'"
    with pytest.raises(AttributeError, match=msg):
        data.apply(transform, axis=1)
@pytest.mark.parametrize(
    "df, func, expected",
    # 使用测试辅助函数获取 Cython 表格参数
    tm.get_cython_table_params(
        DataFrame([["a", "b"], ["b", "a"]]), [["cumprod", TypeError]]
    ),
)
def test_agg_cython_table_raises_frame(df, func, expected, axis, using_infer_string):
    # GH 21224
    # 如果使用推断字符串
    if using_infer_string:
        # 导入 pyarrow 库
        import pyarrow as pa

        # 期望的异常类型更新为 (expected, pa.lib.ArrowNotImplementedError)
        expected = (expected, pa.lib.ArrowNotImplementedError)

    # 错误消息
    msg = "can't multiply sequence by non-int of type 'str'|has no kernel"
    # 如果 func 是字符串类型，则警告为 None；否则为 FutureWarning
    warn = None if isinstance(func, str) else FutureWarning
    # 使用 pytest 检查是否抛出期望异常和匹配消息
    with pytest.raises(expected, match=msg):
        # 使用 tm.assert_produces_warning 检查是否产生警告，匹配特定的警告消息
        with tm.assert_produces_warning(warn, match="using DataFrame.cumprod"):
            # 对 DataFrame 应用聚合函数 func，指定轴向为 axis
            df.agg(func, axis=axis)


@pytest.mark.parametrize(
    "series, func, expected",
    # 使用 chain 函数将多个序列参数连接在一起
    chain(
        tm.get_cython_table_params(
            Series("a b c".split()),
            [
                ("mean", TypeError),  # mean 函数引发 TypeError 异常
                ("prod", TypeError),
                ("std", TypeError),
                ("var", TypeError),
                ("median", TypeError),
                ("cumprod", TypeError),
            ],
        )
    ),
)
def test_agg_cython_table_raises_series(series, func, expected, using_infer_string):
    # GH21224
    # 定义错误消息的正则表达式模式
    msg = r"[Cc]ould not convert|can't multiply sequence by non-int of type"
    # 如果 func 是 "median" 或者是 numpy 的中位数函数，则更新错误消息
    if func == "median" or func is np.nanmedian or func is np.median:
        msg = r"Cannot convert \['a' 'b' 'c'\] to numeric"

    # 如果使用推断字符串
    if using_infer_string:
        # 导入 pyarrow 库
        import pyarrow as pa

        # 期望的异常类型更新为 (expected, pa.lib.ArrowNotImplementedError)
        expected = (expected, pa.lib.ArrowNotImplementedError)

    # 更新错误消息，添加更多匹配模式
    msg = msg + "|does not support|has no kernel"
    # 如果 func 是字符串类型，则警告为 None；否则为 FutureWarning
    warn = None if isinstance(func, str) else FutureWarning

    # 使用 pytest 检查是否抛出期望异常和匹配消息
    with pytest.raises(expected, match=msg):
        # 使用 tm.assert_produces_warning 检查是否产生警告，匹配特定的警告消息
        with tm.assert_produces_warning(warn, match="is currently using Series.*"):
            # 对 Series 应用聚合函数 func
            series.agg(func)


def test_agg_none_to_type():
    # GH 40543
    # 创建包含空值的 DataFrame
    df = DataFrame({"a": [None]})
    # 错误消息使用 re.escape 转义
    msg = re.escape("int() argument must be a string")
    # 使用 pytest 检查是否抛出 TypeError 异常，并匹配消息
    with pytest.raises(TypeError, match=msg):
        # 对 DataFrame 应用转换函数，将空值转换为整数类型
        df.agg({"a": lambda x: int(x.iloc[0])})


def test_transform_none_to_type():
    # GH#34377
    # 创建包含空值的 DataFrame
    df = DataFrame({"a": [None]})
    # 错误消息
    msg = "argument must be a"
    # 使用 pytest 检查是否抛出 TypeError 异常，并匹配消息
    with pytest.raises(TypeError, match=msg):
        # 对 DataFrame 应用 transform 函数，将空值转换为整数类型
        df.transform({"a": lambda x: int(x.iloc[0])})


@pytest.mark.parametrize(
    "func",
    [
        lambda x: np.array([1, 2]).reshape(-1, 2),
        lambda x: [1, 2],
        lambda x: Series([1, 2]),
    ],
)
def test_apply_broadcast_error(func):
    # 创建一个包含重复数组的 DataFrame
    df = DataFrame(
        np.tile(np.arange(3, dtype="int64"), 6).reshape(6, -1) + 1,
        columns=["A", "B", "C"],
    )

    # 错误消息
    msg = "too many dims to broadcast|cannot broadcast result"
    # 使用 pytest 检查是否抛出 ValueError 异常，并匹配消息
    with pytest.raises(ValueError, match=msg):
        # 对 DataFrame 应用 apply 函数，指定轴向为 1，结果类型为 broadcast
        df.apply(func, axis=1, result_type="broadcast")


def test_transform_and_agg_err_agg(axis, float_frame):
    # GH 34377
    # 错误消息
    msg = "cannot combine transform and aggregation operations"
    # 使用 pytest 框架来测试是否抛出指定的 ValueError 异常，并匹配异常消息
    with pytest.raises(ValueError, match=msg):
        # 进入上下文管理器，忽略 NumPy 的所有错误状态，如警告或异常
        with np.errstate(all="ignore"):
            # 对 float_frame 进行聚合操作，计算指定轴上的最大值和平方根
            float_frame.agg(["max", "sqrt"], axis=axis)
# 标记此测试函数忽略特定的 FutureWarning
@pytest.mark.filterwarnings("ignore::FutureWarning")  # GH53325
# 参数化测试函数，用于测试在组合转换和聚合操作时的异常情况
@pytest.mark.parametrize(
    "func, msg",
    [
        # 当 func 包含 "sqrt" 和 "max" 时，期望抛出 ValueError 异常，异常信息为 "cannot combine transform and aggregation"
        (["sqrt", "max"], "cannot combine transform and aggregation"),
        # 当 func 包含 {"foo": np.sqrt, "bar": "sum"} 时，期望抛出 ValueError 异常，异常信息为 "cannot perform both aggregation and transformation"
        (
            {"foo": np.sqrt, "bar": "sum"},
            "cannot perform both aggregation and transformation",
        ),
    ],
)
def test_transform_and_agg_err_series(string_series, func, msg):
    # 在 string_series 上尝试执行 func 操作，验证是否抛出了预期的 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        with np.errstate(all="ignore"):
            string_series.agg(func)


# 参数化测试函数，用于测试在组合转换和聚合操作时的异常情况
@pytest.mark.parametrize("func", [["max", "min"], ["max", "sqrt"]])
def test_transform_wont_agg_frame(axis, float_frame, func):
    # GH 35964
    # 在 float_frame 上尝试执行 func 操作，验证是否抛出了预期的 ValueError 异常，异常信息为 "Function did not transform"
    msg = "Function did not transform"
    with pytest.raises(ValueError, match=msg):
        float_frame.transform(func, axis=axis)


# 参数化测试函数，用于测试在组合转换和聚合操作时的异常情况
@pytest.mark.parametrize("func", [["min", "max"], ["sqrt", "max"]])
def test_transform_wont_agg_series(string_series, func):
    # GH 35964
    # 在 string_series 上尝试执行 func 操作，验证是否抛出了预期的 ValueError 异常，异常信息为 "Function did not transform"
    msg = "Function did not transform"

    with pytest.raises(ValueError, match=msg):
        string_series.transform(func)


# 参数化测试函数，用于测试在组合转换和聚合操作时的异常情况
@pytest.mark.parametrize(
    "op_wrapper", [lambda x: x, lambda x: [x], lambda x: {"A": x}, lambda x: {"A": [x]}]
)
def test_transform_reducer_raises(all_reductions, frame_or_series, op_wrapper):
    # GH 35964
    # 使用 op_wrapper 函数包装 all_reductions 数据，尝试在 frame_or_series 上执行 transform 操作，验证是否抛出了预期的 ValueError 异常，异常信息为 "Function did not transform"
    op = op_wrapper(all_reductions)

    obj = DataFrame({"A": [1, 2, 3]})
    obj = tm.get_obj(obj, frame_or_series)

    msg = "Function did not transform"
    with pytest.raises(ValueError, match=msg):
        obj.transform(op)


# 测试在数据框架中缺少标签时是否引发 KeyError 异常
def test_transform_missing_labels_raises():
    # GH 58474
    # 创建一个 DataFrame df，包含两列 "foo" 和 "bar"，以及索引 ["A", "B", "C"]
    df = DataFrame({"foo": [2, 4, 6], "bar": [1, 2, 3]}, index=["A", "B", "C"])
    # 定义预期的 KeyError 异常信息，验证在 axis=0（行）时是否引发了预期异常
    msg = r"Label\(s\) \['A', 'B'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        # 在行轴上尝试使用指定的转换函数对不存在的标签进行转换
        df.transform({"A": lambda x: x + 2, "B": lambda x: x * 2}, axis=0)

    # 定义预期的 KeyError 异常信息，验证在 axis=1（列）时是否引发了预期异常
    msg = r"Label\(s\) \['bar', 'foo'\] do not exist"
    with pytest.raises(KeyError, match=msg):
        # 在列轴上尝试使用指定的转换函数对不存在的标签进行转换
        df.transform({"foo": lambda x: x + 2, "bar": lambda x: x * 2}, axis=1)
```