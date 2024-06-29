# `D:\src\scipysrc\pandas\pandas\tests\frame\methods\test_replace.py`

```
# 从未来导入类型提示，允许在类型提示中使用字符串字面量
from __future__ import annotations

# 导入 datetime 模块中的 datetime 类
from datetime import datetime
# 导入 re 模块，支持正则表达式操作
import re

# 导入 numpy 库并使用别名 np
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas._config 模块导入 using_pyarrow_string_dtype
from pandas._config import using_pyarrow_string_dtype

# 导入 pandas 库并使用别名 pd
import pandas as pd
# 从 pandas 中导入特定对象：DataFrame, Index, Series, Timestamp, date_range
from pandas import (
    DataFrame,
    Index,
    Series,
    Timestamp,
    date_range,
)
# 导入 pandas._testing 模块并使用别名 tm
import pandas._testing as tm


# 定义 pytest fixture 'mix_ab'，返回一个字典，其值为包含 int 或 str 类型的列表
@pytest.fixture
def mix_ab() -> dict[str, list[int | str]]:
    return {"a": list(range(4)), "b": list("ab..")}


# 定义 pytest fixture 'mix_abc'，返回一个字典，其值为包含 float 或 str 类型的列表
@pytest.fixture
def mix_abc() -> dict[str, list[float | str]]:
    return {"a": list(range(4)), "b": list("ab.."), "c": ["a", "b", np.nan, "d"]}


# TestDataFrameReplace 类定义开始
class TestDataFrameReplace:
    # 标记测试为预期失败，原因为无法将 float 类型设置为 string 类型
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    # 定义测试方法 test_replace_inplace，接受 datetime_frame 和 float_string_frame 两个参数
    def test_replace_inplace(self, datetime_frame, float_string_frame):
        # 将 datetime_frame 中前 5 行的 "A" 列设置为 NaN
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        # 将 datetime_frame 中后 5 行的 "A" 列设置为 NaN
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        # 创建 tsframe 变量作为 datetime_frame 的副本
        tsframe = datetime_frame.copy()
        # 使用 replace 方法将 tsframe 中的 NaN 替换为 0，inplace=True 表示在原地修改
        return_value = tsframe.replace(np.nan, 0, inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 使用 assert_frame_equal 方法比较 tsframe 和 datetime_frame 填充 NaN 为 0 后的结果
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

        # 对于 mixed type 的处理
        mf = float_string_frame
        # 将 float_string_frame 中第 5 到 20 行的 "foo" 列设置为 NaN
        mf.iloc[5:20, mf.columns.get_loc("foo")] = np.nan
        # 将 float_string_frame 中倒数第 10 到最后的行的 "A" 列设置为 NaN
        mf.iloc[-10:, mf.columns.get_loc("A")] = np.nan

        # 使用 replace 方法将 float_string_frame 中的 NaN 替换为 0
        result = float_string_frame.replace(np.nan, 0)
        # 用 fillna 方法填充 float_string_frame 中的 NaN 为 0，预期结果为 expected
        expected = float_string_frame.fillna(value=0)
        # 使用 assert_frame_equal 方法比较 result 和 expected
        tm.assert_frame_equal(result, expected)

        # 创建 tsframe 变量作为 datetime_frame 的副本
        tsframe = datetime_frame.copy()
        # 使用 replace 方法将 tsframe 中的 [NaN] 替换为 [0]，inplace=True 表示在原地修改
        return_value = tsframe.replace([np.nan], [0], inplace=True)
        # 断言返回值为 None
        assert return_value is None
        # 使用 assert_frame_equal 方法比较 tsframe 和 datetime_frame 填充 NaN 为 0 后的结果
        tm.assert_frame_equal(tsframe, datetime_frame.fillna(0))

    # 标记参数化测试，指定参数化变量和期望值
    @pytest.mark.parametrize(
        "to_replace,values,expected",
        [
            # 正则表达式列表和值列表
            # 将 [re1, re2, ..., reN] 替换为 [v1, v2, ..., vN]
            (
                [r"\s*\.\s*", r"e|f|g"],
                [np.nan, "crap"],
                {
                    "a": ["a", "b", np.nan, np.nan],
                    "b": ["crap"] * 3 + ["h"],
                    "c": ["h", "crap", "l", "o"],
                },
            ),
            # 将 [re1, re2, ..., reN] 替换为 [\1\1, \1_crap]
            (
                [r"\s*(\.)\s*", r"(e|f|g)"],
                [r"\1\1", r"\1_crap"],
                {
                    "a": ["a", "b", "..", ".."],
                    "b": ["e_crap", "f_crap", "g_crap", "h"],
                    "c": ["h", "e_crap", "l", "o"],
                },
            ),
            # 将 [re1, re2, ..., reN] 替换为 [(re1 or v1), (re2 or v2), ..., (reN or vN)]
            (
                [r"\s*(\.)\s*", r"e"],
                [r"\1\1", r"crap"],
                {
                    "a": ["a", "b", "..", ".."],
                    "b": ["crap", "f", "g", "h"],
                    "c": ["h", "crap", "l", "o"],
                },
            ),
        ],
    )
    # 标记参数化测试，测试 inplace 参数为 True 和 False 的情况
    @pytest.mark.parametrize("inplace", [True, False])
    # 使用 pytest 的参数化装饰器标记该测试函数，用于多次运行测试用例
    @pytest.mark.parametrize("use_value_regex_args", [True, False])
    # 测试正则表达式替换列表对象的函数
    def test_regex_replace_list_obj(
        self, to_replace, values, expected, inplace, use_value_regex_args
    ):
        # 创建一个 DataFrame 对象，包含三列数据，每列数据由字符列表组成
        df = DataFrame({"a": list("ab.."), "b": list("efgh"), "c": list("helo")})
    
        # 根据 use_value_regex_args 参数的值选择使用不同的参数调用 DataFrame 的 replace 方法
        if use_value_regex_args:
            result = df.replace(value=values, regex=to_replace, inplace=inplace)
        else:
            result = df.replace(to_replace, values, regex=True, inplace=inplace)
    
        # 如果 inplace 参数为 True，则断言结果为 None，并将 df 赋值给 result
        if inplace:
            assert result is None
            result = df
    
        # 创建一个预期的 DataFrame 对象，并使用 assert_frame_equal 函数进行断言比较
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)
    
    # 测试正则表达式替换混合列表的函数
    def test_regex_replace_list_mixed(self, mix_ab):
        # 创建一个混合数据的 DataFrame 对象
        dfmix = DataFrame(mix_ab)
    
        # 定义正则表达式和值的列表，对 dfmix2 执行替换操作，并与预期结果进行比较
        to_replace_res = [r"\s*\.\s*", r"a"]
        values = [np.nan, "crap"]
        mix2 = {"a": list(range(4)), "b": list("ab.."), "c": list("halo")}
        dfmix2 = DataFrame(mix2)
        res = dfmix2.replace(to_replace_res, values, regex=True)
        expec = DataFrame(
            {
                "a": mix2["a"],
                "b": ["crap", "b", np.nan, np.nan],
                "c": ["h", "crap", "l", "o"],
            }
        )
        tm.assert_frame_equal(res, expec)
    
        # 定义正则表达式和值的列表，对 dfmix 执行替换操作，并与预期结果进行比较
        to_replace_res = [r"\s*(\.)\s*", r"(a|b)"]
        values = [r"\1\1", r"\1_crap"]
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({"a": mix_ab["a"], "b": ["a_crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)
    
        # 定义正则表达式和值的列表，对 dfmix 执行替换操作，并与预期结果进行比较
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.replace(to_replace_res, values, regex=True)
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)
    
        # 使用 value 和 regex 参数替换操作，对 dfmix 执行替换操作，并与预期结果进行比较
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        res = dfmix.replace(regex=to_replace_res, value=values)
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        tm.assert_frame_equal(res, expec)
    # 定义一个测试方法，用于测试正则表达式替换列表在原地替换的功能
    def test_regex_replace_list_mixed_inplace(self, mix_ab):
        # 用 mix_ab 创建一个 DataFrame 对象 dfmix
        dfmix = DataFrame(mix_ab)
        
        # 定义需要替换的正则表达式列表和对应的替换值列表
        to_replace_res = [r"\s*\.\s*", r"a"]
        values = [np.nan, "crap"]
        
        # 将 dfmix 复制给 res，并使用 inplace=True 进行原地替换操作，regex=True 表示启用正则表达式替换
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        
        # 使用断言验证返回值为 None
        assert return_value is None
        
        # 创建期望的 DataFrame 对象 expec，用于与 res 进行比较
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b", np.nan, np.nan]})
        
        # 使用 tm.assert_frame_equal 函数比较 res 和 expec，确认其相等
        tm.assert_frame_equal(res, expec)
        
        # 定义另一组正则表达式列表和对应的替换值列表
        to_replace_res = [r"\s*(\.)\s*", r"(a|b)"]
        values = [r"\1\1", r"\1_crap"]
        
        # 将 dfmix 复制给 res，并使用 inplace=True 进行原地替换操作，regex=True 表示启用正则表达式替换
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        
        # 使用断言验证返回值为 None
        assert return_value is None
        
        # 创建期望的 DataFrame 对象 expec，用于与 res 进行比较
        expec = DataFrame({"a": mix_ab["a"], "b": ["a_crap", "b_crap", "..", ".."]})
        
        # 使用 tm.assert_frame_equal 函数比较 res 和 expec，确认其相等
        tm.assert_frame_equal(res, expec)
        
        # 定义另一组正则表达式列表和对应的替换值列表
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        
        # 将 dfmix 复制给 res，并使用 inplace=True 进行原地替换操作，regex=True 表示启用正则表达式替换
        res = dfmix.copy()
        return_value = res.replace(to_replace_res, values, inplace=True, regex=True)
        
        # 使用断言验证返回值为 None
        assert return_value is None
        
        # 创建期望的 DataFrame 对象 expec，用于与 res 进行比较
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        
        # 使用 tm.assert_frame_equal 函数比较 res 和 expec，确认其相等
        tm.assert_frame_equal(res, expec)
        
        # 重复之前的替换操作，但是通过关键字参数传递正则表达式列表和替换值列表
        to_replace_res = [r"\s*(\.)\s*", r"a", r"(b)"]
        values = [r"\1\1", r"crap", r"\1_crap"]
        
        # 将 dfmix 复制给 res，并使用 inplace=True 进行原地替换操作，regex=True 表示启用正则表达式替换
        res = dfmix.copy()
        return_value = res.replace(regex=to_replace_res, value=values, inplace=True)
        
        # 使用断言验证返回值为 None
        assert return_value is None
        
        # 创建期望的 DataFrame 对象 expec，用于与 res 进行比较
        expec = DataFrame({"a": mix_ab["a"], "b": ["crap", "b_crap", "..", ".."]})
        
        # 使用 tm.assert_frame_equal 函数比较 res 和 expec，确认其相等
        tm.assert_frame_equal(res, expec)
    def test_regex_replace_dict_mixed(self, mix_abc):
        dfmix = DataFrame(mix_abc)

        # dicts
        # 单个字典 {re1: v1}，在整个数据框中进行搜索和替换
        res = dfmix.replace({"b": r"\s*\.\s*"}, {"b": np.nan}, regex=True)

        # 复制数据框以备后用
        res2 = dfmix.copy()

        # 使用 inplace=True 替换数据框中的值，并验证返回值为 None
        return_value = res2.replace(
            {"b": r"\s*\.\s*"}, {"b": np.nan}, inplace=True, regex=True
        )
        assert return_value is None

        # 预期的结果数据框
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )

        # 使用测试工具函数验证结果数据框是否与预期相同
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        # 多个字典 {re1: re11, re2: re12, ..., reN: re1N}，在整个数据框中进行搜索和替换
        res = dfmix.replace({"b": r"\s*(\.)\s*"}, {"b": r"\1ty"}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(
            {"b": r"\s*(\.)\s*"}, {"b": r"\1ty"}, inplace=True, regex=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", ".ty", ".ty"], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        # 使用 regex 和 value 参数进行替换
        res = dfmix.replace(regex={"b": r"\s*(\.)\s*"}, value={"b": r"\1ty"})
        res2 = dfmix.copy()
        return_value = res2.replace(
            regex={"b": r"\s*(\.)\s*"}, value={"b": r"\1ty"}, inplace=True
        )
        assert return_value is None
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", ".ty", ".ty"], "c": mix_abc["c"]}
        )
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        # 标量 -> 字典
        # 使用 regex 替换，{value: value}
        expec = DataFrame(
            {"a": mix_abc["a"], "b": [np.nan, "b", ".", "."], "c": mix_abc["c"]}
        )
        res = dfmix.replace("a", {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace("a", {"b": np.nan}, regex=True, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)

        res = dfmix.replace("a", {"b": np.nan}, regex=True)
        res2 = dfmix.copy()
        return_value = res2.replace(regex="a", value={"b": np.nan}, inplace=True)
        assert return_value is None
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
    # 定义一个测试方法，用于测试正则表达式替换嵌套字典的功能
    def test_regex_replace_dict_nested(self, mix_abc):
        # 将混合数据转换为DataFrame对象
        dfmix = DataFrame(mix_abc)
        # 使用正则表达式替换DataFrame中'b'列的特定模式，将匹配的内容替换为NaN
        res = dfmix.replace({"b": {r"\s*\.\s*": np.nan}}, regex=True)
        # 复制DataFrame对象，准备进行下一轮替换
        res2 = dfmix.copy()
        # 复制DataFrame对象，准备进行下一轮替换
        res4 = dfmix.copy()
        # 使用inplace=True参数，将正则表达式替换应用到DataFrame的'b'列，原地修改数据
        return_value = res2.replace(
            {"b": {r"\s*\.\s*": np.nan}}, inplace=True, regex=True
        )
        # 断言原地修改的返回值为None
        assert return_value is None
        # 使用regex参数，对DataFrame的'b'列应用正则表达式替换
        res3 = dfmix.replace(regex={"b": {r"\s*\.\s*": np.nan}})
        # 使用inplace=True参数，将正则表达式替换应用到DataFrame的'b'列，原地修改数据
        return_value = res4.replace(regex={"b": {r"\s*\.\s*": np.nan}}, inplace=True)
        # 断言原地修改的返回值为None
        assert return_value is None
        # 构建预期的DataFrame对象，用于后续断言比较
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )
        # 使用assert_frame_equal方法比较DataFrame对象res与预期对象expec的内容是否相同
        tm.assert_frame_equal(res, expec)
        # 使用assert_frame_equal方法比较DataFrame对象res2与预期对象expec的内容是否相同
        tm.assert_frame_equal(res2, expec)
        # 使用assert_frame_equal方法比较DataFrame对象res3与预期对象expec的内容是否相同
        tm.assert_frame_equal(res3, expec)
        # 使用assert_frame_equal方法比较DataFrame对象res4与预期对象expec的内容是否相同
        tm.assert_frame_equal(res4, expec)

    # 定义一个测试方法，用于测试正则表达式替换的特定情况
    def test_regex_replace_dict_nested_non_first_character(
        self, any_string_dtype, using_infer_string
    ):
        # GH 25259
        # 创建包含字符串数据的DataFrame对象
        dtype = any_string_dtype
        df = DataFrame({"first": ["abc", "bca", "cab"]}, dtype=dtype)
        # 使用正则表达式替换DataFrame中的字符串内容
        result = df.replace({"a": "."}, regex=True)
        # 创建预期的DataFrame对象，用于后续断言比较
        expected = DataFrame({"first": [".bc", "bc.", "c.b"]}, dtype=dtype)
        # 使用assert_frame_equal方法比较DataFrame对象result与预期对象expected的内容是否相同
        tm.assert_frame_equal(result, expected)

    # 使用xfail装饰器标记的测试方法，测试正则表达式替换的特定情况
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    def test_regex_replace_dict_nested_gh4115(self):
        # 创建包含字符串和整数数据的DataFrame对象
        df = DataFrame({"Type": ["Q", "T", "Q", "Q", "T"], "tmp": 2})
        # 创建预期的DataFrame对象，用于后续断言比较
        expected = DataFrame(
            {"Type": Series([0, 1, 0, 0, 1], dtype=df.Type.dtype), "tmp": 2}
        )
        # 使用正则表达式替换DataFrame中的数据
        result = df.replace({"Type": {"Q": 0, "T": 1}})
        # 使用assert_frame_equal方法比较DataFrame对象result与预期对象expected的内容是否相同
        tm.assert_frame_equal(result, expected)

    # 使用xfail装饰器标记的测试方法，测试正则表达式替换的特定情况
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    def test_regex_replace_list_to_scalar(self, mix_abc):
        # 创建包含混合数据的DataFrame对象
        df = DataFrame(mix_abc)
        # 创建预期的DataFrame对象，用于后续断言比较
        expec = DataFrame(
            {
                "a": mix_abc["a"],
                "b": np.array([np.nan] * 4, dtype=object),
                "c": [np.nan, np.nan, np.nan, "d"],
            }
        )

        # 使用正则表达式替换DataFrame中的数据，将匹配的内容替换为NaN
        res = df.replace([r"\s*\.\s*", "a|b"], np.nan, regex=True)
        # 复制DataFrame对象，准备进行下一轮替换
        res2 = df.copy()
        # 复制DataFrame对象，准备进行下一轮替换
        res3 = df.copy()
        # 使用inplace=True参数，将正则表达式替换应用到DataFrame中，原地修改数据
        return_value = res2.replace(
            [r"\s*\.\s*", "a|b"], np.nan, regex=True, inplace=True
        )
        # 断言原地修改的返回值为None
        assert return_value is None
        # 使用inplace=True参数，将正则表达式替换应用到DataFrame中，原地修改数据
        return_value = res3.replace(
            regex=[r"\s*\.\s*", "a|b"], value=np.nan, inplace=True
        )
        # 断言原地修改的返回值为None
        assert return_value is None
        # 使用assert_frame_equal方法比较DataFrame对象res与预期对象expec的内容是否相同
        tm.assert_frame_equal(res, expec)
        # 使用assert_frame_equal方法比较DataFrame对象res2与预期对象expec的内容是否相同
        tm.assert_frame_equal(res2, expec)
        # 使用assert_frame_equal方法比较DataFrame对象res3与预期对象expec的内容是否相同
        tm.assert_frame_equal(res3, expec)
    # 测试用例：使用正则表达式替换字符串到数字
    def test_regex_replace_str_to_numeric(self, mix_abc):
        # 创建数据框对象，载入混合数据
        df = DataFrame(mix_abc)
        # 使用正则表达式替换数据框中的字符串". "为数字0，返回新的数据框
        res = df.replace(r"\s*\.\s*", 0, regex=True)
        # 创建数据框对象的副本
        res2 = df.copy()
        # 在原地修改数据框中的字符串". "为数字0，返回值为None
        return_value = res2.replace(r"\s*\.\s*", 0, inplace=True, regex=True)
        assert return_value is None
        # 创建数据框对象的副本
        res3 = df.copy()
        # 在原地修改数据框中匹配正则表达式"\s*\.\s*"的字符串为数字0，返回值为None
        return_value = res3.replace(regex=r"\s*\.\s*", value=0, inplace=True)
        assert return_value is None
        # 创建预期结果的数据框对象，包含更新后的列"b"
        expec = DataFrame({"a": mix_abc["a"], "b": ["a", "b", 0, 0], "c": mix_abc["c"]})
        # 断言三个数据框对象是否相等
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    # 测试用例：使用正则表达式列表替换字符串到数字
    def test_regex_replace_regex_list_to_numeric(self, mix_abc):
        # 创建数据框对象，载入混合数据
        df = DataFrame(mix_abc)
        # 使用正则表达式列表替换数据框中的字符串". "和"b"为数字0，返回新的数据框
        res = df.replace([r"\s*\.\s*", "b"], 0, regex=True)
        # 创建数据框对象的副本
        res2 = df.copy()
        # 在原地修改数据框中的字符串". "和"b"为数字0，返回值为None
        return_value = res2.replace([r"\s*\.\s*", "b"], 0, regex=True, inplace=True)
        assert return_value is None
        # 创建数据框对象的副本
        res3 = df.copy()
        # 在原地修改数据框中匹配正则表达式列表["\s*\.\s*", "b"]的字符串为数字0，返回值为None
        return_value = res3.replace(regex=[r"\s*\.\s*", "b"], value=0, inplace=True)
        assert return_value is None
        # 创建预期结果的数据框对象，包含更新后的列"b"和"c"
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", 0, 0, 0], "c": ["a", 0, np.nan, "d"]}
        )
        # 断言三个数据框对象是否相等
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    # 测试用例：使用正则表达式替换数据系列中的正则表达式
    def test_regex_replace_series_of_regexes(self, mix_abc):
        # 创建数据框对象，载入混合数据
        df = DataFrame(mix_abc)
        # 创建包含单一值的数据系列对象，用于指定替换的正则表达式
        s1 = Series({"b": r"\s*\.\s*"})
        # 创建包含单一值的数据系列对象，用于指定替换后的值
        s2 = Series({"b": np.nan})
        # 使用正则表达式替换数据框中匹配正则表达式s1的字符串为s2的值，返回新的数据框
        res = df.replace(s1, s2, regex=True)
        # 创建数据框对象的副本
        res2 = df.copy()
        # 在原地修改数据框中匹配正则表达式s1的字符串为s2的值，返回值为None
        return_value = res2.replace(s1, s2, inplace=True, regex=True)
        assert return_value is None
        # 创建数据框对象的副本
        res3 = df.copy()
        # 在原地修改数据框中匹配正则表达式s1的字符串为s2的值，返回值为None
        return_value = res3.replace(regex=s1, value=s2, inplace=True)
        assert return_value is None
        # 创建预期结果的数据框对象，包含更新后的列"b"
        expec = DataFrame(
            {"a": mix_abc["a"], "b": ["a", "b", np.nan, np.nan], "c": mix_abc["c"]}
        )
        # 断言三个数据框对象是否相等
        tm.assert_frame_equal(res, expec)
        tm.assert_frame_equal(res2, expec)
        tm.assert_frame_equal(res3, expec)

    # 测试用例：将数字转换为对象类型
    def test_regex_replace_numeric_to_object_conversion(self, mix_abc):
        # 创建数据框对象，载入混合数据
        df = DataFrame(mix_abc)
        # 创建预期结果的数据框对象，将数据框中的数字0替换为字符串"a"
        expec = DataFrame({"a": ["a", 1, 2, 3], "b": mix_abc["b"], "c": mix_abc["c"]})
        # 使用替换函数将数据框中的数字0替换为字符串"a"，返回新的数据框
        res = df.replace(0, "a")
        # 断言数据框对象是否与预期结果相等
        tm.assert_frame_equal(res, expec)
        # 断言替换后数据框列"a"的数据类型为np.object_
        assert res.a.dtype == np.object_

    @pytest.mark.parametrize(
        "to_replace", [{"": np.nan, ",": ""}, {",": "", "": np.nan}]
    )
    # 定义测试方法，用于测试简单替换和正则表达式替换的联合效果
    def test_joint_simple_replace_and_regex_replace(self, to_replace):
        # 创建一个包含列"col1", "col2", "col3"的DataFrame对象
        df = DataFrame(
            {
                "col1": ["1,000", "a", "3"],
                "col2": ["a", "", "b"],
                "col3": ["a", "b", "c"],
            }
        )
        # 对DataFrame对象进行替换操作，使用传入的正则表达式to_replace
        result = df.replace(regex=to_replace)
        # 创建一个期望的DataFrame对象，以验证替换操作的结果
        expected = DataFrame(
            {
                "col1": ["1000", "a", "3"],
                "col2": ["a", np.nan, "b"],
                "col3": ["a", "b", "c"],
            }
        )
        # 使用测试工具tm.assert_frame_equal来比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    # 使用pytest的参数化装饰器标记，测试正则表达式特殊字符的替换行为
    @pytest.mark.parametrize("metachar", ["[]", "()", r"\d", r"\w", r"\s"])
    def test_replace_regex_metachar(self, metachar):
        # 创建一个包含列"a"的DataFrame对象，其中包含一个特殊字符metachar
        df = DataFrame({"a": [metachar, "else"]})
        # 对DataFrame对象进行替换操作，替换包含特殊字符metachar的值为"paren"
        result = df.replace({"a": {metachar: "paren"}})
        # 创建一个期望的DataFrame对象，以验证替换操作的结果
        expected = DataFrame({"a": ["paren", "else"]})
        # 使用测试工具tm.assert_frame_equal来比较实际结果和期望结果
        tm.assert_frame_equal(result, expected)

    # 使用pytest的参数化装饰器标记，测试字符串类型的正则表达式替换
    @pytest.mark.parametrize(
        "data,to_replace,expected",
        [
            (["xax", "xbx"], {"a": "c", "b": "d"}, ["xcx", "xdx"]),
            (["d", "", ""], {r"^\s*$": pd.NA}, ["d", pd.NA, pd.NA]),
        ],
    )
    def test_regex_replace_string_types(
        self,
        data,
        to_replace,
        expected,
        frame_or_series,
        any_string_dtype,
        using_infer_string,
        request,
    ):
        # GH-41333, GH-35977
        # 根据不同的数据类型进行替换测试，包括空字符串和正则表达式
        dtype = any_string_dtype
        # 根据传入的数据和数据类型创建DataFrame或Series对象
        obj = frame_or_series(data, dtype=dtype)
        # 对DataFrame或Series对象进行替换操作，使用传入的to_replace字典和正则表达式选项
        result = obj.replace(to_replace, regex=True)
        # 根据期望结果和数据类型创建期望的DataFrame或Series对象
        expected = frame_or_series(expected, dtype=dtype)

        # 使用测试工具tm.assert_equal来比较实际结果和期望结果
        tm.assert_equal(result, expected)

    # 测试DataFrame对象的常规替换操作
    def test_replace(self, datetime_frame):
        # 将DataFrame对象的"A"列的前五行和后五行设置为NaN
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan

        # 使用-1e8替换DataFrame对象中的NaN值，生成一个新的DataFrame
        zero_filled = datetime_frame.replace(np.nan, -1e8)
        # 使用测试工具tm.assert_frame_equal比较零填充后的结果和用-1e8填充NaN的结果
        tm.assert_frame_equal(zero_filled, datetime_frame.fillna(-1e8))
        # 使用-1e8替换DataFrame对象中的值为NaN的项，期望还原为原始DataFrame对象
        tm.assert_frame_equal(zero_filled.replace(-1e8, np.nan), datetime_frame)

        # 将DataFrame对象的前五行"A"列设置为NaN，并将前五行"B"列设置为-1e8
        datetime_frame.loc[datetime_frame.index[:5], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[-5:], "A"] = np.nan
        datetime_frame.loc[datetime_frame.index[:5], "B"] = -1e8

        # 创建一个空的DataFrame对象，具有索引为["a", "b"]的行
        df = DataFrame(index=["a", "b"])
        # 使用测试工具tm.assert_frame_equal比较空DataFrame对象和使用replace方法替换5为7后的结果
        tm.assert_frame_equal(df, df.replace(5, 7))

        # GH 11698
        # 测试混合数据类型的情况
        # 创建一个包含元组的DataFrame对象，包含字符串"-"和日期时间对象
        df = DataFrame(
            [("-", pd.to_datetime("20150101")), ("a", pd.to_datetime("20150102"))]
        )
        # 使用np.nan替换DataFrame对象中的"-"字符
        df1 = df.replace("-", np.nan)
        # 创建一个期望的DataFrame对象，验证替换操作的结果
        expected_df = DataFrame(
            [(np.nan, pd.to_datetime("20150101")), ("a", pd.to_datetime("20150102"))]
        )
        # 使用测试工具tm.assert_frame_equal来比较实际结果和期望结果
        tm.assert_frame_equal(df1, expected_df)
    # 定义一个测试方法，用于测试 DataFrame 中的值替换操作
    def test_replace_list(self):
        # 创建一个字典对象，包含三个键值对，每个值为一个字符列表
        obj = {"a": list("ab.."), "b": list("efgh"), "c": list("helo")}
        # 将字典对象转换为 DataFrame 格式
        dfobj = DataFrame(obj)

        # 定义两个列表，分别存储需要替换的正则表达式和替换后的值
        # list of [v1, v2, ..., vN] -> [v1, v2, ..., vN]
        to_replace_res = [r".", r"e"]
        values = [np.nan, "crap"]
        # 使用 DataFrame 的 replace 方法进行替换操作，将结果存储在 res 变量中
        res = dfobj.replace(to_replace_res, values)
        # 创建一个期望的 DataFrame 对象，包含预期的替换结果
        expec = DataFrame(
            {
                "a": ["a", "b", np.nan, np.nan],
                "b": ["crap", "f", "g", "h"],
                "c": ["h", "crap", "l", "o"],
            }
        )
        # 使用 pytest 框架的 assert_frame_equal 方法检查 res 和 expec 是否相等
        tm.assert_frame_equal(res, expec)

        # 再次定义两个列表，进行第二组替换操作
        # list of [v1, v2, ..., vN] -> [v1, v2, .., vN]
        to_replace_res = [r".", r"f"]
        values = [r"..", r"crap"]
        # 再次使用 DataFrame 的 replace 方法进行替换，存储结果在 res 变量中
        res = dfobj.replace(to_replace_res, values)
        # 创建第二组预期的 DataFrame 对象
        expec = DataFrame(
            {
                "a": ["a", "b", "..", ".."],
                "b": ["e", "crap", "g", "h"],
                "c": ["h", "e", "l", "o"],
            }
        )
        # 使用 pytest 框架的 assert_frame_equal 方法检查 res 和 expec 是否相等
        tm.assert_frame_equal(res, expec)

    # 定义另一个测试方法，用于测试空列表作为替换值的情况
    def test_replace_with_empty_list(self, frame_or_series):
        # GH 21977
        # 创建一个 Series 对象，包含包含列表的值
        ser = Series([["a", "b"], [], np.nan, [1]])
        # 创建一个 DataFrame 对象，列名为 'col'，值为 ser
        obj = DataFrame({"col": ser})
        # 调用 tm.get_obj 方法获取处理后的 obj 对象
        obj = tm.get_obj(obj, frame_or_series)
        # 将期望结果设置为 obj 的副本
        expected = obj
        # 使用 replace 方法将空列表 [] 替换为 np.nan，存储结果在 result 变量中
        result = obj.replace([], np.nan)
        # 使用 pytest 框架的 assert_equal 方法检查 result 和 expected 是否相等
        tm.assert_equal(result, expected)

        # GH 19266
        # 创建一条错误消息，用于 pytest 的异常检查
        msg = (
            "NumPy boolean array indexing assignment cannot assign {size} "
            "input values to the 1 output values where the mask is true"
        )
        # 使用 pytest 的 raises 方法检查是否会抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg.format(size=0)):
            obj.replace({np.nan: []})
        # 再次使用 raises 方法检查是否会抛出 ValueError 异常，并匹配特定的错误消息
        with pytest.raises(ValueError, match=msg.format(size=2)):
            obj.replace({np.nan: ["dummy", "alt"]})

    # 定义另一个测试方法，用于测试 Series 替换为字典的情况
    def test_replace_series_dict(self):
        # from GH 3064
        # 创建一个 DataFrame 对象，包含两列 'zero' 和 'one'，每列为一个字典
        df = DataFrame({"zero": {"a": 0.0, "b": 1}, "one": {"a": 2.0, "b": 0}})
        # 使用 replace 方法将值为 0 替换为字典形式的值，存储结果在 result 变量中
        result = df.replace(0, {"zero": 0.5, "one": 1.0})
        # 创建一个期望的 DataFrame 对象，包含预期的替换结果
        expected = DataFrame({"zero": {"a": 0.5, "b": 1}, "one": {"a": 2.0, "b": 1.0}})
        # 使用 pytest 框架的 assert_frame_equal 方法检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 再次使用 replace 方法，将值为 0 替换为 df 的均值，存储结果在 result 变量中
        result = df.replace(0, df.mean())
        # 使用 pytest 框架的 assert_frame_equal 方法检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # series to series/dict
        # 创建一个新的 DataFrame 对象，包含两列 'zero' 和 'one'，每列为一个字典
        df = DataFrame({"zero": {"a": 0.0, "b": 1}, "one": {"a": 2.0, "b": 0}})
        # 创建一个 Series 对象，包含两个键 'zero' 和 'one'，对应值为 0.0 和 2.0
        s = Series({"zero": 0.0, "one": 2.0})
        # 使用 replace 方法将 Series 对象 s 替换为字典形式的值，存储结果在 result 变量中
        result = df.replace(s, {"zero": 0.5, "one": 1.0})
        # 创建一个期望的 DataFrame 对象，包含预期的替换结果
        expected = DataFrame({"zero": {"a": 0.5, "b": 1}, "one": {"a": 1.0, "b": 0.0}})
        # 使用 pytest 框架的 assert_frame_equal 方法检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 再次使用 replace 方法，将 Series 对象 s 替换为 df 的均值，存储结果在 result 变量中
        result = df.replace(s, df.mean())
        # 使用 pytest 框架的 assert_frame_equal 方法检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.xfail 标记的测试用例，测试特定条件下的预期失败情况
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    # 定义一个测试函数，用于测试数据帧的替换功能
    def test_replace_convert(self):
        # 创建一个包含两行三列的数据帧，用于测试
        df = DataFrame([["foo", "bar", "bah"], ["bar", "foo", "bah"]])
        # 创建一个替换映射，将字符串替换为整数
        m = {"foo": 1, "bar": 2, "bah": 3}
        # 使用替换映射对数据帧进行替换操作
        rep = df.replace(m)
        # 期望的结果是原始数据帧的数据类型
        expec = df.dtypes
        # 实际结果是替换后数据帧的数据类型
        res = rep.dtypes
        # 断言期望结果和实际结果相等
        tm.assert_series_equal(expec, res)

    # 使用 pytest 的 mark 来标记这个测试为预期失败，原因是不能将浮点数设置为字符串
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    # 定义一个测试函数，测试混合类型数据帧的替换功能
    def test_replace_mixed(self, float_string_frame):
        # 获取一个包含浮点数和字符串的混合类型数据帧
        mf = float_string_frame
        # 将特定位置的元素设置为 NaN
        mf.iloc[5:20, mf.columns.get_loc("foo")] = np.nan
        mf.iloc[-10:, mf.columns.get_loc("A")] = np.nan

        # 将数据帧中的 NaN 替换为 -18
        result = float_string_frame.replace(np.nan, -18)
        # 期望的结果是将 NaN 替换为 -18 后的数据帧
        expected = float_string_frame.fillna(value=-18)
        # 断言替换后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)
        # 再次将 -18 替换回 NaN，期望结果是原始的混合类型数据帧
        tm.assert_frame_equal(result.replace(-18, np.nan), float_string_frame)

        # 将数据帧中的 NaN 替换为 -1e8
        result = float_string_frame.replace(np.nan, -1e8)
        # 期望的结果是将 NaN 替换为 -1e8 后的数据帧
        expected = float_string_frame.fillna(value=-1e8)
        # 断言替换后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)
        # 再次将 -1e8 替换回 NaN，期望结果是原始的混合类型数据帧
        tm.assert_frame_equal(result.replace(-1e8, np.nan), float_string_frame)

    # 定义一个测试函数，测试替换整型数据块并进行类型转换
    def test_replace_mixed_int_block_upcasting(self):
        # 创建一个包含浮点数和整数的数据帧
        df = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0, 1], dtype="int64"),
            }
        )
        # 期望的结果是将整数 0 替换为浮点数 0.5 后的数据帧
        expected = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0.5, 1], dtype="float64"),
            }
        )
        # 将数据帧中的整数 0 替换为浮点数 0.5
        result = df.replace(0, 0.5)
        # 断言替换后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)

        # 在 inplace 模式下将数据帧中的整数 0 替换为浮点数 0.5
        return_value = df.replace(0, 0.5, inplace=True)
        # 断言 inplace 替换的返回值为 None
        assert return_value is None
        # 断言原始数据帧与期望的数据帧相等
        tm.assert_frame_equal(df, expected)

    # 定义一个测试函数，测试替换整型数据块并进行类型分割
    def test_replace_mixed_int_block_splitting(self):
        # 创建一个包含浮点数和整数的数据帧
        df = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0, 1], dtype="int64"),
                "C": Series([1, 2], dtype="int64"),
            }
        )
        # 期望的结果是将整数 0 替换为浮点数 0.5 后的数据帧
        expected = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0.5, 1], dtype="float64"),
                "C": Series([1, 2], dtype="int64"),
            }
        )
        # 将数据帧中的整数 0 替换为浮点数 0.5
        result = df.replace(0, 0.5)
        # 断言替换后的结果与期望结果相等
        tm.assert_frame_equal(result, expected)
    def test_replace_mixed2(self, using_infer_string):
        # 定义一个包含浮点和整数列的数据帧对象
        df = DataFrame(
            {
                "A": Series([1.0, 2.0], dtype="float64"),
                "B": Series([0, 1], dtype="int64"),
            }
        )
        # 创建预期结果的数据帧对象，其中"A"列包含了一个字符串对象
        expected = DataFrame(
            {
                "A": Series([1, "foo"], dtype="object"),
                "B": Series([0, 1], dtype="int64"),
            }
        )
        # 使用 replace 方法替换数据帧中的特定值，并将结果存储在变量 result 中
        result = df.replace(2, "foo")
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(result, expected)

        # 创建预期结果的数据帧对象，其中"A"列和"B"列包含了字符串对象
        expected = DataFrame(
            {
                "A": Series(["foo", "bar"]),
                "B": Series([0, "foo"], dtype="object"),
            }
        )
        # 使用 replace 方法替换数据帧中的特定值，并将结果存储在变量 result 中
        result = df.replace([1, 2], ["foo", "bar"])
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(result, expected)

    def test_replace_mixed3(self):
        # 来自测试案例的数据帧对象创建
        df = DataFrame(
            {"A": Series([3, 0], dtype="int64"), "B": Series([0, 3], dtype="int64")}
        )
        # 使用均值创建一个字典，并替换数据帧中的特定值
        result = df.replace(3, df.mean().to_dict())
        # 创建一个预期结果的数据帧对象，其数据类型为 float64
        expected = df.copy().astype("float64")
        # 计算数据帧中每列的均值
        m = df.mean()
        # 更新预期结果的特定元素，使其等于相应列的均值
        expected.iloc[0, 0] = m.iloc[0]
        expected.iloc[1, 1] = m.iloc[1]
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(result, expected)

    def test_replace_nullable_int_with_string_doesnt_cast(self):
        # GH#25438: 不将 df['a'] 强制转换为 float64 类型
        df = DataFrame({"a": [1, 2, 3, np.nan], "b": ["some", "strings", "here", "he"]})
        # 将 "a" 列转换为 Nullable 整数类型
        df["a"] = df["a"].astype("Int64")

        # 使用 replace 方法替换空字符串为 NaN
        res = df.replace("", np.nan)
        # 使用断言函数检查结果列 "a" 是否等于原始列 "a"
        tm.assert_series_equal(res["a"], df["a"])

    @pytest.mark.parametrize("dtype", ["boolean", "Int64", "Float64"])
    def test_replace_with_nullable_column(self, dtype):
        # GH-44499: 可空列的替换
        # 创建一个可空序列对象，其数据类型由参数 dtype 指定
        nullable_ser = Series([1, 0, 1], dtype=dtype)
        # 创建一个包含可空列的数据帧对象
        df = DataFrame({"A": ["A", "B", "x"], "B": nullable_ser})
        # 使用 replace 方法替换数据帧中的特定值，并将结果存储在变量 result 中
        result = df.replace("x", "X")
        # 创建预期结果的数据帧对象
        expected = DataFrame({"A": ["A", "B", "X"], "B": nullable_ser})
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(result, expected)

    def test_replace_simple_nested_dict(self):
        # 创建一个简单的数据帧对象
        df = DataFrame({"col": range(1, 5)})
        # 创建预期结果的数据帧对象，其中 "col" 列的部分元素被替换为字符串
        expected = DataFrame({"col": ["a", 2, 3, "b"]})

        # 使用 replace 方法替换单个嵌套字典中的值，并将结果存储在变量 result 中
        result = df.replace({"col": {1: "a", 4: "b"}})
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(expected, result)

        # 在此情况下，应与非嵌套版本相同
        # 使用 replace 方法替换数据帧中的特定值，并将结果存储在变量 result 中
        result = df.replace({1: "a", 4: "b"})
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(expected, result)

    def test_replace_simple_nested_dict_with_nonexistent_value(self):
        # 创建一个简单的数据帧对象
        df = DataFrame({"col": range(1, 5)})
        # 创建预期结果的数据帧对象，其中 "col" 列的部分元素被替换为字符串
        expected = DataFrame({"col": ["a", 2, 3, "b"]})

        # 使用 replace 方法替换单个嵌套字典中的值，并将结果存储在变量 result 中
        result = df.replace({-1: "-", 1: "a", 4: "b"})
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(expected, result)

        # 使用 replace 方法替换单个嵌套字典中的值，并将结果存储在变量 result 中
        result = df.replace({"col": {-1: "-", 1: "a", 4: "b"}})
        # 使用断言函数检查 result 是否等于预期结果 expected
        tm.assert_frame_equal(expected, result)
    # 定义一个测试函数，用于测试替换 pd.NA 为 None 的行为
    def test_replace_NA_with_None(self):
        # 注释：用于标识 issue gh-45601
        df = DataFrame({"value": [42, None]}).astype({"value": "Int64"})
        # 执行替换操作，将 pd.NA 替换为 None
        result = df.replace({pd.NA: None})
        # 期望的结果 DataFrame
        expected = DataFrame({"value": [42, None]}, dtype=object)
        # 断言结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，用于测试替换 pd.NaT 为 None 的行为
    def test_replace_NAT_with_None(self):
        # 注释：用于标识 issue gh-45836
        df = DataFrame([pd.NaT, pd.NaT])
        # 执行替换操作，将 pd.NaT 和 np.nan 替换为 None
        result = df.replace({pd.NaT: None, np.nan: None})
        # 期望的结果 DataFrame
        expected = DataFrame([None, None])
        # 断言结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试替换特定值为 None 时保持分类数据类型不变的行为
    def test_replace_with_None_keeps_categorical(self):
        # 注释：用于标识 issue gh-46634
        cat_series = Series(["b", "b", "b", "d"], dtype="category")
        df = DataFrame(
            {
                "id": Series([5, 4, 3, 2], dtype="float64"),
                "col": cat_series,
            }
        )
        # 执行替换操作，将值为 3 替换为 None
        result = df.replace({3: None})
        # 期望的结果 DataFrame，保持分类数据类型不变
        expected = DataFrame(
            {
                "id": Series([5.0, 4.0, None, 2.0], dtype="object"),
                "col": cat_series,
            }
        )
        # 断言结果是否与期望一致
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试替换 np.nan 为 0 的行为，并保持结果的一致性
    def test_replace_value_is_none(self, datetime_frame):
        # 记录原始值
        orig_value = datetime_frame.iloc[0, 0]
        orig2 = datetime_frame.iloc[1, 0]

        # 将第一行第一列和第二行第一列置为 np.nan 和 1
        datetime_frame.iloc[0, 0] = np.nan
        datetime_frame.iloc[1, 0] = 1

        # 执行替换操作，将 np.nan 替换为 0
        result = datetime_frame.replace(to_replace={np.nan: 0})
        # 期望的结果，将 np.nan 替换为 0，保持一致性
        expected = datetime_frame.T.replace(to_replace={np.nan: 0}).T
        # 断言结果是否与期望一致
        tm.assert_frame_equal(result, expected)

        # 再次执行替换操作，将 np.nan 替换为 0，将 1 替换为 -1e8
        result = datetime_frame.replace(to_replace={np.nan: 0, 1: -1e8})
        # 构建期望的结果 DataFrame
        tsframe = datetime_frame.copy()
        tsframe.iloc[0, 0] = 0
        tsframe.iloc[1, 0] = -1e8
        expected = tsframe
        # 断言结果是否与期望一致
        tm.assert_frame_equal(expected, result)
        
        # 恢复原始值
        datetime_frame.iloc[0, 0] = orig_value
        datetime_frame.iloc[1, 0] = orig2

    # 定义一个测试函数，测试替换 np.nan 为 -1e8 后的数据类型行为
    def test_replace_for_new_dtypes(self, datetime_frame):
        # 注释：用于标识 dtypes
        tsframe = datetime_frame.copy().astype(np.float32)
        # 将列 "A" 的前五行和后五行置为 np.nan
        tsframe.loc[tsframe.index[:5], "A"] = np.nan
        tsframe.loc[tsframe.index[-5:], "A"] = np.nan

        # 执行替换操作，将 np.nan 替换为 -1e8
        zero_filled = tsframe.replace(np.nan, -1e8)
        # 断言替换为 -1e8 的结果与使用 fillna(-1e8) 的结果一致
        tm.assert_frame_equal(zero_filled, tsframe.fillna(-1e8))
        # 再次执行替换操作，将 -1e8 替换为 np.nan，验证是否与原始 tsframe 一致
        tm.assert_frame_equal(zero_filled.replace(-1e8, np.nan), tsframe)

        # 将列 "A" 的前五行和后五行置为 np.nan
        tsframe.loc[tsframe.index[:5], "A"] = np.nan
        tsframe.loc[tsframe.index[-5:], "A"] = np.nan
        # 将列 "B" 的前五行置为 np.nan
        tsframe.loc[tsframe.index[:5], "B"] = np.nan
    def test_replace_input_formats_listlike(self):
        # 定义要替换的字典
        to_rep = {"A": np.nan, "B": 0, "C": ""}
        # 定义替换后的值的字典
        values = {"A": 0, "B": -1, "C": "missing"}
        # 创建一个包含指定数据的 DataFrame 对象
        df = DataFrame(
            {"A": [np.nan, 0, np.inf], "B": [0, 2, 5], "C": ["", "asdf", "fd"]}
        )
        # 使用指定的字典进行替换，并返回新的 DataFrame 对象
        filled = df.replace(to_rep, values)
        # 生成期望的替换结果字典
        expected = {k: v.replace(to_rep[k], values[k]) for k, v in df.items()}
        # 比较填充后的 DataFrame 和期望的 DataFrame 是否相等
        tm.assert_frame_equal(filled, DataFrame(expected))

        # 使用列表对列表进行替换
        result = df.replace([0, 2, 5], [5, 2, 0])
        # 生成期望的替换结果 DataFrame
        expected = DataFrame(
            {"A": [np.nan, 5, np.inf], "B": [5, 2, 0], "C": ["", "asdf", "fd"]}
        )
        # 比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 标量替换为字典
        values = {"A": 0, "B": -1, "C": "missing"}
        # 创建一个包含指定数据的 DataFrame 对象
        df = DataFrame(
            {"A": [np.nan, 0, np.nan], "B": [0, 2, 5], "C": ["", "asdf", "fd"]}
        )
        # 使用指定的标量进行替换，并返回新的 DataFrame 对象
        filled = df.replace(np.nan, values)
        # 生成期望的替换结果字典
        expected = {k: v.replace(np.nan, values[k]) for k, v in df.items()}
        # 比较填充后的 DataFrame 和期望的 DataFrame 是否相等
        tm.assert_frame_equal(filled, DataFrame(expected))

        # 使用列表对列表进行替换
        to_rep = [np.nan, 0, ""]
        values = [-2, -1, "missing"]
        # 使用指定的列表进行替换，并返回新的 DataFrame 对象
        result = df.replace(to_rep, values)
        # 复制原始 DataFrame 作为期望的 DataFrame
        expected = df.copy()
        # 遍历并替换期望的 DataFrame
        for rep, value in zip(to_rep, values):
            return_value = expected.replace(rep, value, inplace=True)
            assert return_value is None
        # 比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 检查替换列表长度不匹配的情况
        msg = r"Replacement lists must match in length\. Expecting 3 got 2"
        with pytest.raises(ValueError, match=msg):
            df.replace(to_rep, values[1:])

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    def test_replace_input_formats_scalar(self):
        df = DataFrame(
            {"A": [np.nan, 0, np.inf], "B": [0, 2, 5], "C": ["", "asdf", "fd"]}
        )

        # 字典替换为标量
        to_rep = {"A": np.nan, "B": 0, "C": ""}
        filled = df.replace(to_rep, 0)
        # 生成期望的替换结果字典
        expected = {k: v.replace(to_rep[k], 0) for k, v in df.items()}
        # 比较填充后的 DataFrame 和期望的 DataFrame 是否相等
        tm.assert_frame_equal(filled, DataFrame(expected))

        # 检查非法的值替换类型
        msg = "value argument must be scalar, dict, or Series"
        with pytest.raises(TypeError, match=msg):
            df.replace(to_rep, [np.nan, 0, ""])

        # 使用列表对标量进行替换
        to_rep = [np.nan, 0, ""]
        # 使用指定的列表进行替换，并返回新的 DataFrame 对象
        result = df.replace(to_rep, -1)
        # 复制原始 DataFrame 作为期望的 DataFrame
        expected = df.copy()
        # 遍历并替换期望的 DataFrame
        for rep in to_rep:
            return_value = expected.replace(rep, -1, inplace=True)
            assert return_value is None
        # 比较结果和期望的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_replace_limit(self):
        # TODO
        pass

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    # 定义一个测试函数，用于测试不使用正则表达式进行字典替换的情况
    def test_replace_dict_no_regex(self):
        # 创建一个 Series 对象，包含从数字到对应文本的映射
        answer = Series(
            {
                0: "Strongly Agree",
                1: "Agree",
                2: "Neutral",
                3: "Disagree",
                4: "Strongly Disagree",
            }
        )
        # 创建一个字典，将文本映射到权重值
        weights = {
            "Agree": 4,
            "Disagree": 2,
            "Neutral": 3,
            "Strongly Agree": 5,
            "Strongly Disagree": 1,
        }
        # 创建一个期望的 Series 对象，将原始文本映射到对应的权重值
        expected = Series({0: 5, 1: 4, 2: 3, 3: 2, 4: 1}, dtype=answer.dtype)
        # 对 answer 中的文本进行权重替换，生成结果 Series 对象
        result = answer.replace(weights)
        # 使用测试工具检查结果 Series 是否与期望的 Series 相等
        tm.assert_series_equal(result, expected)

    # 使用 pytest 的标记 xfail 标记一个预期失败的测试
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    # 定义一个测试函数，测试不使用正则表达式进行 Series 替换的情况
    def test_replace_series_no_regex(self):
        # 创建一个 Series 对象，包含从数字到对应文本的映射
        answer = Series(
            {
                0: "Strongly Agree",
                1: "Agree",
                2: "Neutral",
                3: "Disagree",
                4: "Strongly Disagree",
            }
        )
        # 创建一个 Series 对象，将文本映射到权重值
        weights = Series(
            {
                "Agree": 4,
                "Disagree": 2,
                "Neutral": 3,
                "Strongly Agree": 5,
                "Strongly Disagree": 1,
            }
        )
        # 创建一个期望的 Series 对象，将原始文本映射到对应的权重值
        expected = Series({0: 5, 1: 4, 2: 3, 3: 2, 4: 1}, dtype=object)
        # 对 answer 中的文本进行权重替换，生成结果 Series 对象
        result = answer.replace(weights)
        # 使用测试工具检查结果 Series 是否与期望的 Series 相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，测试字典替换中元组和列表的顺序不变
    def test_replace_dict_tuple_list_ordering_remains_the_same(self):
        # 创建一个 DataFrame 对象，包含列 A，其中包含 NaN 和 1
        df = DataFrame({"A": [np.nan, 1]})
        # 使用字典进行替换，将 NaN 替换为 0，将 1 替换为 -1e8
        res1 = df.replace(to_replace={np.nan: 0, 1: -1e8})
        # 使用元组进行替换，将 1 替换为 -1e8，将 NaN 替换为 0
        res2 = df.replace(to_replace=(1, np.nan), value=[-1e8, 0])
        # 使用列表进行替换，将 1 替换为 -1e8，将 NaN 替换为 0
        res3 = df.replace(to_replace=[1, np.nan], value=[-1e8, 0])
        # 创建一个期望的 DataFrame 对象，将列 A 中的 NaN 替换为 0，将 1 替换为 -1e8
        expected = DataFrame({"A": [0, -1e8]})
        # 使用测试工具检查 res1 和 res2 是否相等
        tm.assert_frame_equal(res1, res2)
        # 使用测试工具检查 res2 和 res3 是否相等
        tm.assert_frame_equal(res2, res3)
        # 使用测试工具检查 res3 是否与期望的 DataFrame 相等
        tm.assert_frame_equal(res3, expected)

    # 定义一个测试函数，测试 DataFrame 中不使用正则表达式替换操作
    def test_replace_doesnt_replace_without_regex(self):
        # 创建一个 DataFrame 对象，包含多个列和行
        df = DataFrame(
            {
                "fol": [1, 2, 2, 3],
                "T_opp": ["0", "vr", "0", "0"],
                "T_Dir": ["0", "0", "0", "bt"],
                "T_Enh": ["vo", "0", "0", "0"],
            }
        )
        # 使用字典进行替换，但是由于没有启用正则表达式，不会进行替换操作
        res = df.replace({r"\D": 1})
        # 使用测试工具检查原始 DataFrame 是否与结果 DataFrame 相等
        tm.assert_frame_equal(df, res)

    # 定义一个测试函数，测试将布尔值替换为字符串
    def test_replace_bool_with_string(self):
        # 创建一个 DataFrame 对象，包含布尔列和字符串列
        df = DataFrame({"a": [True, False], "b": list("ab")})
        # 将 True 替换为字符串 "a"
        result = df.replace(True, "a")
        # 创建一个期望的 DataFrame 对象，将 True 替换为 "a"
        expected = DataFrame({"a": ["a", False], "b": df.b})
        # 使用测试工具检查结果 DataFrame 是否与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试不执行任何替换操作的情况
    def test_replace_pure_bool_with_string_no_op(self):
        # 创建一个 DataFrame 对象，包含随机生成的布尔值
        df = DataFrame(np.random.default_rng(2).random((2, 2)) > 0.5)
        # 尝试将所有值 "asdf" 替换为 "fdsa"，但是不会进行替换操作
        result = df.replace("asdf", "fdsa")
        # 使用测试工具检查原始 DataFrame 是否与结果 DataFrame 相等
        tm.assert_frame_equal(df, result)

    # 定义一个测试函数，测试将布尔值替换为布尔值的情况
    def test_replace_bool_with_bool(self):
        # 创建一个 DataFrame 对象，包含随机生成的布尔值
        df = DataFrame(np.random.default_rng(2).random((2, 2)) > 0.5)
        # 将 False 替换为 True
        result = df.replace(False, True)
        # 创建一个期望的 DataFrame 对象，将所有 False 替换为 True
        expected = DataFrame(np.ones((2, 2), dtype=bool))
        # 使用测试工具检查结果 DataFrame 是否与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)
    def test_replace_with_dict_with_bool_keys(self):
        # 创建一个包含布尔键的DataFrame对象
        df = DataFrame({0: [True, False], 1: [False, True]})
        # 使用字典进行替换操作，将 {"asdf": "asdb", True: "yes"} 替换应用到DataFrame中
        result = df.replace({"asdf": "asdb", True: "yes"})
        # 预期的DataFrame对象，用于断言比较
        expected = DataFrame({0: ["yes", False], 1: [False, "yes"]})
        # 使用断言函数确认结果与预期相等
        tm.assert_frame_equal(result, expected)

    def test_replace_dict_strings_vs_ints(self):
        # GH#34789
        # 创建一个包含字符串键的DataFrame对象
        df = DataFrame({"Y0": [1, 2], "Y1": [3, 4]})
        # 使用字典进行替换操作，将 {"replace_string": "test"} 替换应用到DataFrame中
        result = df.replace({"replace_string": "test"})

        # 使用断言函数确认结果与原始DataFrame相等
        tm.assert_frame_equal(result, df)

        # 对DataFrame中的单独列进行替换操作
        result = df["Y0"].replace({"replace_string": "test"})
        # 使用断言函数确认结果与原始列相等
        tm.assert_series_equal(result, df["Y0"])

    def test_replace_truthy(self):
        # 创建一个包含布尔值的DataFrame对象
        df = DataFrame({"a": [True, True]})
        # 替换DataFrame中的 np.inf 和 -np.inf 为 np.nan
        r = df.replace([np.inf, -np.inf], np.nan)
        # 期望的DataFrame对象
        e = df
        # 使用断言函数确认结果与预期相等
        tm.assert_frame_equal(r, e)

    def test_nested_dict_overlapping_keys_replace_int(self):
        # GH 27660 keep behaviour consistent for simple dictionary and
        # nested dictionary replacement
        # 创建一个包含整数列的DataFrame对象
        df = DataFrame({"a": list(range(1, 5))})

        # 使用嵌套字典进行替换操作，将 {"a": {1: 2, 2: 3, 3: 4, 4: 5}} 替换应用到DataFrame中的列 'a'
        result = df.replace({"a": dict(zip(range(1, 5), range(2, 6)))})
        # 期望的DataFrame对象，使用相同的字典进行替换
        expected = df.replace(dict(zip(range(1, 5), range(2, 6))))
        # 使用断言函数确认结果与预期相等
        tm.assert_frame_equal(result, expected)

    def test_nested_dict_overlapping_keys_replace_str(self):
        # GH 27660
        # 创建一个包含字符串列的DataFrame对象
        a = np.arange(1, 5)
        astr = a.astype(str)
        bstr = np.arange(2, 6).astype(str)
        df = DataFrame({"a": astr})
        # 使用字典进行替换操作，将 {"a": {"1": "2", "2": "3", "3": "4", "4": "5"}} 替换应用到DataFrame中的列 'a'
        result = df.replace(dict(zip(astr, bstr)))
        # 期望的DataFrame对象，使用相同的嵌套字典进行替换
        expected = df.replace({"a": dict(zip(astr, bstr))})
        # 使用断言函数确认结果与预期相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    def test_replace_swapping_bug(self, using_infer_string):
        # 创建一个包含布尔值列的DataFrame对象
        df = DataFrame({"a": [True, False, True]})
        # 使用嵌套字典进行替换操作，将 {"a": {True: "Y", False: "N"}} 替换应用到DataFrame中的列 'a'
        res = df.replace({"a": {True: "Y", False: "N"}})
        # 期望的DataFrame对象
        expect = DataFrame({"a": ["Y", "N", "Y"]})
        # 使用断言函数确认结果与预期相等
        tm.assert_frame_equal(res, expect)

        # 创建一个包含整数列的DataFrame对象
        df = DataFrame({"a": [0, 1, 0]})
        # 使用嵌套字典进行替换操作，将 {"a": {0: "Y", 1: "N"}} 替换应用到DataFrame中的列 'a'
        res = df.replace({"a": {0: "Y", 1: "N"}})
        # 期望的DataFrame对象
        expect = DataFrame({"a": ["Y", "N", "Y"]})
        # 使用断言函数确认结果与预期相等
        tm.assert_frame_equal(res, expect)
    # 定义测试函数，用于测试处理带有时区的 datetime64[ns, tz] 数据
    def test_replace_datetimetz(self):
        # GH 11326: GitHub issue number for reference
        # 当处理 datetime64[ns, tz] 数据时，出现异常行为的问题
        # 创建一个 DataFrame，包含带有时区信息的日期范围和数值列
        df = DataFrame(
            {
                "A": date_range("20130101", periods=3, tz="US/Eastern"),
                "B": [0, np.nan, 2],
            }
        )
        # 使用 replace 方法将 NaN 替换为 1，生成结果 DataFrame
        result = df.replace(np.nan, 1)
        # 创建预期的 DataFrame，将 NaN 替换为 1 后的期望结果
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=3, tz="US/Eastern"),
                "B": Series([0, 1, 2], dtype="float64"),
            }
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 fillna 方法将 NaN 替换为 1，生成结果 DataFrame
        result = df.fillna(1)
        # 断言填充 NaN 后的 DataFrame 是否与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 使用 replace 方法将 0 替换为 NaN，生成结果 DataFrame
        result = df.replace(0, np.nan)
        # 创建预期的 DataFrame，将 0 替换为 NaN 后的期望结果
        expected = DataFrame(
            {
                "A": date_range("20130101", periods=3, tz="US/Eastern"),
                "B": [np.nan, np.nan, 2],
            }
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 replace 方法将指定的时间戳替换为另一个时间戳，生成结果 DataFrame
        result = df.replace(
            Timestamp("20130102", tz="US/Eastern"),
            Timestamp("20130104", tz="US/Eastern"),
        )
        # 创建预期的 DataFrame，将指定时间戳替换后的期望结果
        expected = DataFrame(
            {
                "A": [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130104", tz="US/Eastern"),
                    Timestamp("20130103", tz="US/Eastern"),
                ],
                "B": [0, np.nan, 2],
            }
        )
        # 将预期结果的时间戳转换为纳秒单位
        expected["A"] = expected["A"].dt.as_unit("ns")
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 复制原始 DataFrame
        result = df.copy()
        # 将指定位置的值设为 NaN
        result.iloc[1, 0] = np.nan
        # 使用 replace 方法将特定值替换为指定时间戳，生成结果 DataFrame
        result = result.replace({"A": pd.NaT}, Timestamp("20130104", tz="US/Eastern"))
        # 断言替换后的 DataFrame 是否与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 在版本 2.0 之前，此操作可能导致对象类型不匹配的问题
        # 复制原始 DataFrame
        result = df.copy()
        # 将指定位置的值设为 NaN
        result.iloc[1, 0] = np.nan
        # 使用 replace 方法将特定值替换为指定时间戳，生成结果 DataFrame
        result = result.replace({"A": pd.NaT}, Timestamp("20130104", tz="US/Pacific"))
        # 创建预期的 DataFrame，将时区转换为 US/Eastern 后的期望结果
        expected = DataFrame(
            {
                "A": [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130104", tz="US/Pacific").tz_convert("US/Eastern"),
                    Timestamp("20130103", tz="US/Eastern"),
                ],
                "B": [0, np.nan, 2],
            }
        )
        # 将预期结果的时间戳转换为纳秒单位
        expected["A"] = expected["A"].dt.as_unit("ns")
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 复制原始 DataFrame
        result = df.copy()
        # 将指定位置的值设为 NaN
        result.iloc[1, 0] = np.nan
        # 使用 replace 方法将 NaN 替换为指定时间戳，生成结果 DataFrame
        result = result.replace({"A": np.nan}, Timestamp("20130104"))
        # 创建预期的 DataFrame，将指定 NaN 替换为时间戳后的期望结果
        expected = DataFrame(
            {
                "A": [
                    Timestamp("20130101", tz="US/Eastern"),
                    Timestamp("20130104"),
                    Timestamp("20130103", tz="US/Eastern"),
                ],
                "B": [0, np.nan, 2],
            }
        )
        # 断言两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    def test_replace_with_empty_dictlike(self, mix_abc):
        # GH 15289
        # 创建 DataFrame 对象，使用 mix_abc 作为数据
        df = DataFrame(mix_abc)
        # 使用空字典替换 DataFrame 中所有值，并断言替换后的 DataFrame 与原 DataFrame 相等
        tm.assert_frame_equal(df, df.replace({}))
        # 使用空 Series 替换 DataFrame 中所有值，并断言替换后的 DataFrame 与原 DataFrame 相等
        tm.assert_frame_equal(df, df.replace(Series([], dtype=object)))

        # 使用字典 {"b": {}} 替换 DataFrame 中的值，并断言替换后的 DataFrame 与原 DataFrame 相等
        tm.assert_frame_equal(df, df.replace({"b": {}}))
        # 使用 Series {"b": {}} 替换 DataFrame 中的值，并断言替换后的 DataFrame 与原 DataFrame 相等
        tm.assert_frame_equal(df, df.replace(Series({"b": {}})))

    @pytest.mark.parametrize(
        "df, to_replace, exp",
        [
            (
                {"col1": [1, 2, 3], "col2": [4, 5, 6]},
                {4: 5, 5: 6, 6: 7},
                {"col1": [1, 2, 3], "col2": [5, 6, 7]},
            ),
            (
                {"col1": [1, 2, 3], "col2": ["4", "5", "6"]},
                {"4": "5", "5": "6", "6": "7"},
                {"col1": [1, 2, 3], "col2": ["5", "6", "7"]},
            ),
        ],
    )
    def test_replace_commutative(self, df, to_replace, exp):
        # GH 16051
        # DataFrame.replace() 在非数值的情况下进行值的覆盖
        # 当问题出现在 Series 中时，也会添加到数据帧中

        # 创建 DataFrame 对象，使用参数 df 作为数据
        df = DataFrame(df)

        # 创建期望的 DataFrame 对象，使用参数 exp 作为数据
        expected = DataFrame(exp)
        # 使用 to_replace 字典替换 DataFrame 中的值，并断言替换后的 DataFrame 与期望的 DataFrame 相等
        result = df.replace(to_replace)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    @pytest.mark.parametrize(
        "replacer",
        [
            Timestamp("20170827"),
            np.int8(1),
            np.int16(1),
            np.float32(1),
            np.float64(1),
        ],
    )
    def test_replace_replacer_dtype(self, replacer):
        # GH26632
        # 创建 DataFrame 对象，包含单元素列表 ["a"]
        df = DataFrame(["a"])
        # 使用 replacer 替换 DataFrame 中的值，并期望结果为包含 replacer 值的 DataFrame
        result = df.replace({"a": replacer, "b": replacer})
        # 创建期望的 DataFrame 对象，包含单元素列表 [replacer]，数据类型为 object
        expected = DataFrame([replacer], dtype=object)
        # 断言替换后的 DataFrame 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_replace_after_convert_dtypes(self):
        # GH31517
        # 创建 DataFrame 对象，使用字典 {"grp": [1, 2, 3, 4, 5]}，并指定数据类型为 "Int64"
        df = DataFrame({"grp": [1, 2, 3, 4, 5]}, dtype="Int64")
        # 使用 replace() 方法将值 1 替换为 10，并期望结果中 "grp" 列的第一个值为 10
        result = df.replace(1, 10)
        # 创建期望的 DataFrame 对象，使用字典 {"grp": [10, 2, 3, 4, 5]}，并指定数据类型为 "Int64"
        expected = DataFrame({"grp": [10, 2, 3, 4, 5]}, dtype="Int64")
        # 断言替换后的 DataFrame 与期望的 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_replace_invalid_to_replace(self):
        # GH 18634
        # API: replace() 如果提供了无效的参数，应该引发异常
        # 创建 DataFrame 对象，包含两个列 "one" 和 "two"，各包含字符串数据
        df = DataFrame({"one": ["a", "b ", "c"], "two": ["d ", "e ", "f "]})
        # 设置异常消息的正则表达式
        msg = (
            r"Expecting 'to_replace' to be either a scalar, array-like, "
            r"dict or None, got invalid type.*"
        )
        # 使用 pytest 的 raises() 方法断言替换操作会引发 TypeError，并匹配指定的异常消息
        with pytest.raises(TypeError, match=msg):
            df.replace(lambda x: x.strip())

    @pytest.mark.parametrize("dtype", ["float", "float64", "int64", "Int64", "boolean"])
    @pytest.mark.parametrize("value", [np.nan, pd.NA])
    def test_replace_no_replacement_dtypes(self, dtype, value):
        # https://github.com/pandas-dev/pandas/issues/32988
        # 创建 DataFrame 对象，使用单位矩阵作为数据，并指定数据类型为参数 dtype
        df = DataFrame(np.eye(2), dtype=dtype)
        # 使用 replace() 方法将 None、-np.inf 和 np.inf 替换为指定的 value 值，并期望结果与原 DataFrame 相等
        result = df.replace(to_replace=[None, -np.inf, np.inf], value=value)
        # 断言替换后的 DataFrame 与原 DataFrame 相等
        tm.assert_frame_equal(result, df)
    @pytest.mark.parametrize("replacement", [np.nan, 5])
    def test_replace_with_duplicate_columns(self, replacement):
        # 标记此测试用例为参数化测试，参数为 replacement，分别测试 np.nan 和 5
        # GH 24798：关联的 GitHub issue 编号
        result = DataFrame({"A": [1, 2, 3], "A1": [4, 5, 6], "B": [7, 8, 9]})
        # 创建 DataFrame result 包含列 A, A1, B，及其对应的数据

        result.columns = list("AAB")
        # 重置 result 的列名为 ["A", "A", "B"]

        expected = DataFrame(
            {"A": [1, 2, 3], "A1": [4, 5, 6], "B": [replacement, 8, 9]}
        )
        # 创建期望的 DataFrame expected，替换 B 列的第一个元素为 replacement

        expected.columns = list("AAB")
        # 重置 expected 的列名为 ["A", "A", "B"]

        result["B"] = result["B"].replace(7, replacement)
        # 替换 result 中 B 列的值为 7 的元素为 replacement

        tm.assert_frame_equal(result, expected)
        # 使用测试工具 tm 来比较 result 和 expected 的内容是否相同

    @pytest.mark.parametrize("value", [pd.Period("2020-01"), pd.Interval(0, 5)])
    def test_replace_ea_ignore_float(self, frame_or_series, value):
        # 标记此测试用例为参数化测试，参数为 value，分别测试 pd.Period("2020-01") 和 pd.Interval(0, 5)
        # GH#34871：关联的 GitHub issue 编号
        obj = DataFrame({"Per": [value] * 3})
        # 创建 DataFrame obj，包含 Per 列，每行值都是 value 的副本，共三行

        obj = tm.get_obj(obj, frame_or_series)
        # 使用测试工具 tm 处理 obj，将其转换为 frame_or_series 格式

        expected = obj.copy()
        # 复制 obj 作为 expected 的期望结果

        result = obj.replace(1.0, 0.0)
        # 替换 obj 中的值为 1.0 的元素为 0.0，生成 result

        tm.assert_equal(expected, result)
        # 使用测试工具 tm 来比较 expected 和 result 的内容是否相同

    @pytest.mark.parametrize(
        "replace_dict, final_data",
        [({"a": 1, "b": 1}, [[2, 2], [2, 2]]), ({"a": 1, "b": 2}, [[2, 1], [2, 2]])],
    )
    def test_categorical_replace_with_dict(self, replace_dict, final_data):
        # 标记此测试用例为参数化测试，参数为 replace_dict 和 final_data
        # GH 26988：关联的 GitHub issue 编号
        df = DataFrame([[1, 1], [2, 2]], columns=["a", "b"], dtype="category")
        # 创建具有两列 'a' 和 'b' 的 DataFrame df，数据类型为 category

        final_data = np.array(final_data)
        # 将 final_data 转换为 NumPy 数组

        a = pd.Categorical(final_data[:, 0], categories=[1, 2])
        # 创建分类变量 a，使用 final_data 第一列的值，并指定可能的分类 [1, 2]

        b = pd.Categorical(final_data[:, 1], categories=[1, 2])
        # 创建分类变量 b，使用 final_data 第二列的值，并指定可能的分类 [1, 2]

        expected = DataFrame({"a": a, "b": b})
        # 创建期望的 DataFrame expected，包含分类变量 a 和 b

        result = df.replace(replace_dict, 2)
        # 使用 replace_dict 将 df 中的值替换为 2，生成 result

        tm.assert_frame_equal(result, expected)
        # 使用测试工具 tm 来比较 result 和 expected 的内容是否相同

        msg = r"DataFrame.iloc\[:, 0\] \(column name=\"a\"\) are " "different"
        # 设置错误消息的正则表达式字符串

        with pytest.raises(AssertionError, match=msg):
            # 检查是否会引发 AssertionError，且错误消息与 msg 匹配
            # 确保非原地调用不会影响原始 DataFrame
            tm.assert_frame_equal(df, expected)

        return_value = df.replace(replace_dict, 2, inplace=True)
        # 原地替换 df 中的值为 2，返回值为 None

        assert return_value is None
        # 断言原地替换的返回值为 None

        tm.assert_frame_equal(df, expected)
        # 使用测试工具 tm 来比较 df 和 expected 的内容是否相同
    def test_replace_value_category_type(self):
        """
        Test for #23305: to ensure category dtypes are maintained
        after replace with direct values
        """

        # create input data
        input_dict = {
            "col1": [1, 2, 3, 4],  # 创建一个包含整数的列表作为输入数据的字典的值
            "col2": ["a", "b", "c", "d"],  # 创建一个包含字符串的列表作为输入数据的字典的值
            "col3": [1.5, 2.5, 3.5, 4.5],  # 创建一个包含浮点数的列表作为输入数据的字典的值
            "col4": ["cat1", "cat2", "cat3", "cat4"],  # 创建一个包含字符串的列表作为输入数据的字典的值
            "col5": ["obj1", "obj2", "obj3", "obj4"],  # 创建一个包含字符串的列表作为输入数据的字典的值
        }
        # explicitly cast columns as category and order them
        input_df = DataFrame(data=input_dict).astype(
            {"col2": "category", "col4": "category"}  # 将特定列转换为分类类型，并按特定顺序排列
        )
        input_df["col2"] = input_df["col2"].cat.reorder_categories(
            ["a", "b", "c", "d"], ordered=True  # 重新排列分类列的类别，并设置为有序
        )
        input_df["col4"] = input_df["col4"].cat.reorder_categories(
            ["cat1", "cat2", "cat3", "cat4"], ordered=True  # 重新排列分类列的类别，并设置为有序
        )

        # create expected dataframe
        expected_dict = {
            "col1": [1, 2, 3, 4],  # 创建一个包含整数的列表作为期望数据的字典的值
            "col2": ["a", "b", "c", "z"],  # 创建一个包含字符串的列表作为期望数据的字典的值
            "col3": [1.5, 2.5, 3.5, 4.5],  # 创建一个包含浮点数的列表作为期望数据的字典的值
            "col4": ["cat1", "catX", "cat3", "cat4"],  # 创建一个包含字符串的列表作为期望数据的字典的值
            "col5": ["obj9", "obj2", "obj3", "obj4"],  # 创建一个包含字符串的列表作为期望数据的字典的值
        }
        # explicitly cast columns as category and order them
        expected = DataFrame(data=expected_dict).astype(
            {"col2": "category", "col4": "category"}  # 将特定列转换为分类类型，并按特定顺序排列
        )
        expected["col2"] = expected["col2"].cat.reorder_categories(
            ["a", "b", "c", "z"], ordered=True  # 重新排列分类列的类别，并设置为有序
        )
        expected["col4"] = expected["col4"].cat.reorder_categories(
            ["cat1", "catX", "cat3", "cat4"], ordered=True  # 重新排列分类列的类别，并设置为有序
        )

        # replace values in input dataframe
        input_df = input_df.apply(
            lambda x: x.astype("category").cat.rename_categories({"d": "z"})  # 将指定列中的某个类别重命名
        )
        input_df = input_df.apply(
            lambda x: x.astype("category").cat.rename_categories({"obj1": "obj9"})  # 将指定列中的某个类别重命名
        )
        result = input_df.apply(
            lambda x: x.astype("category").cat.rename_categories({"cat2": "catX"})  # 将指定列中的某个类别重命名
        )

        result = result.astype({"col1": "int64", "col3": "float64", "col5": "object"})  # 将结果数据框的列转换为特定的数据类型
        tm.assert_frame_equal(result, expected)  # 使用测试工具比较两个数据框是否相等
    def test_replace_dict_category_type(self):
        """
        Test to ensure category dtypes are maintained
        after replace with dict values
        """
        # GH#35268, GH#44940

        # create input dataframe
        input_dict = {"col1": ["a"], "col2": ["obj1"], "col3": ["cat1"]}
        # explicitly cast columns as category
        input_df = DataFrame(data=input_dict).astype(
            {"col1": "category", "col2": "category", "col3": "category"}
        )

        # create expected dataframe
        expected_dict = {"col1": ["z"], "col2": ["obj9"], "col3": ["catX"]}
        # explicitly cast columns as category
        expected = DataFrame(data=expected_dict).astype(
            {"col1": "category", "col2": "category", "col3": "category"}
        )

        # replace values in input dataframe using a dict
        result = input_df.apply(
            lambda x: x.cat.rename_categories(
                {"a": "z", "obj1": "obj9", "cat1": "catX"}
            )
        )

        tm.assert_frame_equal(result, expected)

    def test_replace_with_compiled_regex(self):
        # https://github.com/pandas-dev/pandas/issues/35680
        # create dataframe with strings
        df = DataFrame(["a", "b", "c"])
        # compile regex pattern
        regex = re.compile("^a$")
        # replace using compiled regex
        result = df.replace({regex: "z"}, regex=True)
        # create expected dataframe
        expected = DataFrame(["z", "b", "c"])
        tm.assert_frame_equal(result, expected)

    def test_replace_intervals(self):
        # https://github.com/pandas-dev/pandas/issues/35931
        # create dataframe with intervals
        df = DataFrame({"a": [pd.Interval(0, 1), pd.Interval(0, 1)]})
        # replace intervals with values
        result = df.replace({"a": {pd.Interval(0, 1): "x"}})
        # create expected dataframe
        expected = DataFrame({"a": ["x", "x"]})
        tm.assert_frame_equal(result, expected)

    def test_replace_unicode(self):
        # GH: 16784
        # create mapping dictionary for replacement
        columns_values_map = {"positive": {"正面": 1, "中立": 1, "负面": 0}}
        # create dataframe with ones
        df1 = DataFrame({"positive": np.ones(3)})
        # replace values using unicode mapping
        result = df1.replace(columns_values_map)
        # create expected dataframe
        expected = DataFrame({"positive": np.ones(3)})
        tm.assert_frame_equal(result, expected)

    def test_replace_bytes(self, frame_or_series):
        # GH#38900
        # create object from frame or series containing byte-like data
        obj = frame_or_series(["o"]).astype("|S")
        # make a copy of the expected object
        expected = obj.copy()
        # replace None values with NaN
        obj = obj.replace({None: np.nan})
        tm.assert_equal(obj, expected)

    @pytest.mark.parametrize(
        "data, to_replace, value, expected",
        [
            ([1], [1.0], [0], [0]),
            ([1], [1], [0], [0]),
            ([1.0], [1.0], [0], [0.0]),
            ([1.0], [1], [0], [0.0]),
        ],
    )
    @pytest.mark.parametrize("box", [list, tuple, np.array])
    def test_replace_list_with_mixed_type(
        self, data, to_replace, value, expected, box, frame_or_series
    ):
        # GH#40371
        # create object from frame or series with mixed type data
        obj = frame_or_series(data)
        # create expected object from frame or series
        expected = frame_or_series(expected)
        # replace values in object using specified box type
        result = obj.replace(box(to_replace), value)
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("val", [2, np.nan, 2.0])
    # 定义一个测试方法，用于测试替换DataFrame中值为None的情况，传入参数val表示要替换的值
    def test_replace_value_none_dtype_numeric(self, val):
        # GH#48231: GitHub issue编号，指明这段代码修复了什么问题或实现了什么功能
        # 创建一个DataFrame对象，包含列'a'，第一行为1，第二行为val
        df = DataFrame({"a": [1, val]})
        # 使用replace方法将DataFrame中所有val的值替换为None，赋值给result
        result = df.replace(val, None)
        # 创建一个预期的DataFrame对象，包含列'a'，第一行为1，第二行为None，数据类型为object
        expected = DataFrame({"a": [1, None]}, dtype=object)
        # 断言两个DataFrame对象是否相等，使用tm.assert_frame_equal方法
        tm.assert_frame_equal(result, expected)

        # 重新创建DataFrame对象，包含列'a'，第一行为1，第二行为val
        df = DataFrame({"a": [1, val]})
        # 使用replace方法，通过字典将DataFrame中所有val的值替换为None，赋值给result
        result = df.replace({val: None})
        # 断言两个DataFrame对象是否相等，使用tm.assert_frame_equal方法
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试替换DataFrame中特定字符串为pd.NA的情况
    def test_replace_with_nil_na(self):
        # GH 32075: GitHub issue编号，指明这段代码修复了什么问题或实现了什么功能
        # 创建一个DataFrame对象，包含列'a'，第一行为"nil"，第二行为pd.NA
        ser = DataFrame({"a": ["nil", pd.NA]})
        # 创建一个预期的DataFrame对象，包含列'a'，第一行为"anything else"，第二行为pd.NA，指定索引为[0, 1]
        expected = DataFrame({"a": ["anything else", pd.NA]}, index=[0, 1])
        # 使用replace方法将DataFrame中所有"nil"的值替换为"anything else"，赋值给result
        result = ser.replace("nil", "anything else")
        # 断言两个DataFrame对象是否相等，使用tm.assert_frame_equal方法
        tm.assert_frame_equal(expected, result)
# 定义 TestDataFrameReplaceRegex 类，用于测试 DataFrame 中的正则表达式替换功能
class TestDataFrameReplaceRegex:
    
    # 使用 pytest.mark.parametrize 注释装饰器定义参数化测试
    @pytest.mark.parametrize(
        "data",
        [
            {"a": list("ab.."), "b": list("efgh")},  # 第一个数据集，包含键 'a' 和 'b'
            {"a": list("ab.."), "b": list(range(4))},  # 第二个数据集，包含键 'a' 和 'b'
        ],
    )
    # 使用 pytest.mark.parametrize 注释装饰器定义参数化测试
    @pytest.mark.parametrize(
        "to_replace,value", [(r"\s*\.\s*", np.nan), (r"\s*(\.)\s*", r"\1\1\1")]
    )
    # 使用 pytest.mark.parametrize 注释装饰器定义参数化测试
    @pytest.mark.parametrize("compile_regex", [True, False])
    # 使用 pytest.mark.parametrize 注释装饰器定义参数化测试
    @pytest.mark.parametrize("regex_kwarg", [True, False])
    # 使用 pytest.mark.parametrize 注释装饰器定义参数化测试
    @pytest.mark.parametrize("inplace", [True, False])
    # 定义 test_regex_replace_scalar 方法，测试正则表达式替换
    def test_regex_replace_scalar(
        self, data, to_replace, value, compile_regex, regex_kwarg, inplace
    ):
        # 创建 DataFrame 对象，传入 data 字典
        df = DataFrame(data)
        # 复制 DataFrame 对象作为期望的结果
        expected = df.copy()

        # 如果 compile_regex 为 True，则将 to_replace 编译成正则表达式对象
        if compile_regex:
            to_replace = re.compile(to_replace)

        # 如果 regex_kwarg 为 True，则将 to_replace 赋值给 regex 变量，将 to_replace 设为 None
        if regex_kwarg:
            regex = to_replace
            to_replace = None
        else:
            regex = True

        # 执行替换操作，将结果存储在 result 变量中
        result = df.replace(to_replace, value, inplace=inplace, regex=regex)

        # 如果 inplace 为 True，则断言结果为 None，并将 df 赋值给 result
        if inplace:
            assert result is None
            result = df

        # 根据 value 的值，设置期望的替换结果值 expected_replace_val
        if value is np.nan:
            expected_replace_val = np.nan
        else:
            expected_replace_val = "..."

        # 在 expected DataFrame 中，根据条件将 'a' 列中为 '.' 的值替换为 expected_replace_val
        expected.loc[expected["a"] == ".", "a"] = expected_replace_val
        # 使用 assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.xfail 注释装饰器，表示预期这个测试用例会失败
    @pytest.mark.xfail(
        using_pyarrow_string_dtype(), reason="can't set float into string"
    )
    # 使用 pytest.mark.parametrize 注释装饰器定义参数化测试
    @pytest.mark.parametrize("regex", [False, True])
    # 定义 test_replace_regex_dtype_frame 方法，测试替换操作对数据类型的影响
    def test_replace_regex_dtype_frame(self, regex):
        # 创建 DataFrame 对象 df1，包含键 'A' 和 'B'，值为字符串 "0"
        df1 = DataFrame({"A": ["0"], "B": ["0"]})
        # 创建期望的 DataFrame 对象 expected_df1，将 'A' 列中的 "0" 替换为整数 1，并保留原数据类型
        expected_df1 = DataFrame({"A": [1], "B": [1]}, dtype=df1.dtypes.iloc[0])
        # 执行替换操作，将结果存储在 result_df1 变量中
        result_df1 = df1.replace(to_replace="0", value=1, regex=regex)
        # 使用 assert_frame_equal 函数断言 result_df1 和 expected_df1 是否相等
        tm.assert_frame_equal(result_df1, expected_df1)

        # 创建 DataFrame 对象 df2，包含键 'A' 和 'B'，值为字符串 "0" 和 "1"
        df2 = DataFrame({"A": ["0"], "B": ["1"]})
        # 创建期望的 DataFrame 对象 expected_df2，将 'A' 列中的 "0" 替换为整数 1，并保留原数据类型
        expected_df2 = DataFrame({"A": [1], "B": ["1"]}, dtype=df2.dtypes.iloc[0])
        # 执行替换操作，将结果存储在 result_df2 变量中
        result_df2 = df2.replace(to_replace="0", value=1, regex=regex)
        # 使用 assert_frame_equal 函数断言 result_df2 和 expected_df2 是否相等
        tm.assert_frame_equal(result_df2, expected_df2)

    # 定义 test_replace_with_value_also_being_replaced 方法，测试替换操作中替换的值本身也会被替换
    def test_replace_with_value_also_being_replaced(self):
        # 创建 DataFrame 对象 df，包含键 'A' 和 'B'，值为 [0, 1, 2] 和 [1, 0, 2]
        df = DataFrame({"A": [0, 1, 2], "B": [1, 0, 2]})
        # 执行替换操作，将结果存储在 result 变量中
        result = df.replace({0: 1, 1: np.nan})
        # 创建期望的 DataFrame 对象 expected，将 'A' 列中的 0 替换为 1，1 替换为 NaN
        expected = DataFrame({"A": [1, np.nan, 2], "B": [np.nan, 1, 2]})
        # 使用 assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义 test_replace_categorical_no_replacement 方法，测试分类数据中的替换操作不会生效
    def test_replace_categorical_no_replacement(self):
        # 创建 DataFrame 对象 df，包含键 'a' 和 'b'，值为 ['one', 'two', None, 'three'] 和 ['one', None, 'two', 'three']，数据类型为 'category'
        df = DataFrame(
            {
                "a": ["one", "two", None, "three"],
                "b": ["one", None, "two", "three"],
            },
            dtype="category",
        )
        # 复制 DataFrame 对象作为期望的结果
        expected = df.copy()

        # 执行替换操作，将结果存储在 result 变量中
        result = df.replace(to_replace=[".", "def"], value=["_", None])
        # 使用 assert_frame_equal 函数断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    # 定义测试函数，用于测试对象替换功能，同时传入 using_infer_string 参数
    def test_replace_object_splitting(self, using_infer_string):
        # GH#53977
        # 创建一个包含两列的 DataFrame 对象，一列为单个字符串元素，另一列为字符串
        df = DataFrame({"a": ["a"], "b": "b"})
        # 根据 using_infer_string 参数判断条件，验证 DataFrame 内部数据块的数量是否符合预期
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1
        # 使用正则表达式替换 DataFrame 中匹配到的空白字符串为指定值，并在原地进行修改
        df.replace(to_replace=r"^\s*$", value="", inplace=True, regex=True)
        # 再次根据 using_infer_string 参数判断条件，验证 DataFrame 内部数据块的数量是否符合预期
        if using_infer_string:
            assert len(df._mgr.blocks) == 2
        else:
            assert len(df._mgr.blocks) == 1
```