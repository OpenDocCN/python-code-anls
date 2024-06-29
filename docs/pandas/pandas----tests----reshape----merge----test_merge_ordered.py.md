# `D:\src\scipysrc\pandas\pandas\tests\reshape\merge\test_merge_ordered.py`

```
# 导入必要的模块和库
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库，并重命名为 np
import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 Pandas 库，并重命名为 pd
from pandas import (  # 从 Pandas 库中导入特定模块
    DataFrame,  # 导入 DataFrame 类
    merge_ordered,  # 导入 merge_ordered 函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块


@pytest.fixture
def left():
    # 返回一个 DataFrame 对象，包含指定的键和数值
    return DataFrame({"key": ["a", "c", "e"], "lvalue": [1, 2.0, 3]})


@pytest.fixture
def right():
    # 返回一个 DataFrame 对象，包含指定的键和数值
    return DataFrame({"key": ["b", "c", "d", "f"], "rvalue": [1, 2, 3.0, 4]})


class TestMergeOrdered:
    def test_basic(self, left, right):
        # 使用 merge_ordered 函数进行基本的有序合并，基于 'key' 列
        result = merge_ordered(left, right, on="key")
        # 预期结果的 DataFrame 对象，包含 'key', 'lvalue', 'rvalue' 列
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1, np.nan, 2, np.nan, 3, np.nan],
                "rvalue": [np.nan, 1, 2, 3, np.nan, 4],
            }
        )
        # 使用 Pandas 内置的 assert_frame_equal 函数比较两个 DataFrame 对象
        tm.assert_frame_equal(result, expected)

    def test_ffill(self, left, right):
        # 使用 merge_ordered 函数进行有序合并，并使用前向填充方法 'ffill'
        result = merge_ordered(left, right, on="key", fill_method="ffill")
        # 预期结果的 DataFrame 对象，包含 'key', 'lvalue', 'rvalue' 列
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"],
                "lvalue": [1.0, 1, 2, 2, 3, 3.0],
                "rvalue": [np.nan, 1, 2, 3, 3, 4],
            }
        )
        # 使用 Pandas 内置的 assert_frame_equal 函数比较两个 DataFrame 对象
        tm.assert_frame_equal(result, expected)

    def test_multigroup(self, left, right):
        # 将 left DataFrame 重复一次，并添加 'group' 列
        left = pd.concat([left, left], ignore_index=True)
        left["group"] = ["a"] * 3 + ["b"] * 3
        
        # 使用 merge_ordered 函数进行多组有序合并，基于 'key' 列和 'group' 列，使用前向填充方法 'ffill'
        result = merge_ordered(
            left, right, on="key", left_by="group", fill_method="ffill"
        )
        # 预期结果的 DataFrame 对象，包含 'key', 'lvalue', 'rvalue', 'group' 列
        expected = DataFrame(
            {
                "key": ["a", "b", "c", "d", "e", "f"] * 2,
                "lvalue": [1.0, 1, 2, 2, 3, 3.0] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3, 4] * 2,
            }
        )
        expected["group"] = ["a"] * 6 + ["b"] * 6
        
        # 使用 Pandas 内置的 assert_frame_equal 函数比较两个 DataFrame 对象的部分内容
        tm.assert_frame_equal(result, expected.loc[:, result.columns])

        # 使用相反的顺序进行多组有序合并
        result2 = merge_ordered(
            right, left, on="key", right_by="group", fill_method="ffill"
        )
        # 使用 Pandas 内置的 assert_frame_equal 函数比较两个 DataFrame 对象的部分内容
        tm.assert_frame_equal(result, result2.loc[:, result.columns])

        # 使用 merge_ordered 函数进行有序合并，基于 'key' 列和 'group' 列，但不使用填充方法
        result = merge_ordered(left, right, on="key", left_by="group")
        # 使用 assert 语句确保结果中 'group' 列没有缺失值
        assert result["group"].notna().all()

    @pytest.mark.filterwarnings(
        "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
    )
    def test_merge_type(self, left, right):
        # 定义一个继承自 DataFrame 的子类 NotADataFrame
        class NotADataFrame(DataFrame):
            @property
            def _constructor(self):
                return NotADataFrame

        # 使用 NotADataFrame 类初始化一个对象 nad
        nad = NotADataFrame(left)
        # 使用 merge 方法将 nad 和 right DataFrame 对象合并，基于 'key' 列
        result = nad.merge(right, on="key")

        # 使用 assert 语句确认结果对象的类型是 NotADataFrame 类型
        assert isinstance(result, NotADataFrame)

    @pytest.mark.parametrize(
        "df_seq, pattern",
        [
            ((), "[Nn]o objects"),  # 参数化测试用例：空元组，匹配模式 "[Nn]o objects"
            ([], "[Nn]o objects"),  # 参数化测试用例：空列表，匹配模式 "[Nn]o objects"
            ({}, "[Nn]o objects"),  # 参数化测试用例：空字典，匹配模式 "[Nn]o objects"
            ([None], "objects.*None"),  # 参数化测试用例：包含一个 None 元素的列表，匹配模式 "objects.*None"
            ([None, None], "objects.*None"),  # 参数化测试用例：包含两个 None 元素的列表，匹配模式 "objects.*None"
        ],
    )
    def test_empty_sequence_concat(self, df_seq, pattern):
        # GH 9157
        # 使用 pytest.raises 检查在执行 pd.concat(df_seq) 时是否抛出 ValueError 异常，匹配指定模式
        with pytest.raises(ValueError, match=pattern):
            pd.concat(df_seq)
    @pytest.mark.parametrize(
        "arg", [[DataFrame()], [None, DataFrame()], [DataFrame(), None]]
    )
    # 使用 pytest 的参数化装饰器，定义了一个测试函数 test_empty_sequence_concat_ok，测试 pd.concat 对不同参数的行为
    def test_empty_sequence_concat_ok(self, arg):
        pd.concat(arg)

    def test_doc_example(self):
        # 创建一个左侧 DataFrame，包含 "group", "key", "lvalue" 三列
        left = DataFrame(
            {
                "group": list("aaabbb"),
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3] * 2,
            }
        )

        # 创建一个右侧 DataFrame，包含 "key", "rvalue" 两列
        right = DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})

        # 调用 merge_ordered 函数，将 left 和 right 进行按组合并，使用前向填充策略
        result = merge_ordered(left, right, fill_method="ffill", left_by="group")

        # 创建期望的 DataFrame，预期结果
        expected = DataFrame(
            {
                "group": list("aaaaabbbbb"),
                "key": ["a", "b", "c", "d", "e"] * 2,
                "lvalue": [1, 1, 2, 2, 3] * 2,
                "rvalue": [np.nan, 1, 2, 3, 3] * 2,
            }
        )

        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "left, right, on, left_by, right_by, expected",
        [
            (
                {"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]},
                {"T": [2], "E": [1]},
                ["T"],
                ["G", "H"],
                None,
                {
                    "G": ["g"] * 3,
                    "H": ["h"] * 3,
                    "T": [1, 2, 3],
                    "E": [np.nan, 1.0, np.nan],
                },
            ),
            (
                {"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]},
                {"T": [2], "E": [1]},
                "T",
                ["G", "H"],
                None,
                {
                    "G": ["g"] * 3,
                    "H": ["h"] * 3,
                    "T": [1, 2, 3],
                    "E": [np.nan, 1.0, np.nan],
                },
            ),
            (
                {"T": [2], "E": [1]},
                {"G": ["g", "g"], "H": ["h", "h"], "T": [1, 3]},
                ["T"],
                None,
                ["G", "H"],
                {
                    "T": [1, 2, 3],
                    "E": [np.nan, 1.0, np.nan],
                    "G": ["g"] * 3,
                    "H": ["h"] * 3,
                },
            ),
        ],
    )
    # 测试函数 test_list_type_by，验证 merge_ordered 在不同参数组合下的行为是否符合预期
    def test_list_type_by(self, left, right, on, left_by, right_by, expected):
        # GH 35269
        # 将 left 和 right 转换为 DataFrame 对象
        left = DataFrame(left)
        right = DataFrame(right)
        
        # 调用 merge_ordered 函数，将 left 和 right 进行合并
        result = merge_ordered(
            left=left,
            right=right,
            on=on,
            left_by=left_by,
            right_by=right_by,
        )
        
        # 将期望的结果转换为 DataFrame 对象
        expected = DataFrame(expected)

        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    def test_left_by_length_equals_to_right_shape0(self):
        # GH 38166
        # 创建左侧 DataFrame，包含三列数据
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        # 创建右侧 DataFrame，包含两列数据
        right = DataFrame([[2, 1]], columns=list("ET"))
        # 使用 merge_ordered 函数将左右两个 DataFrame 按照列 E 进行有序合并，以列 G 和 H 作为左侧合并的关键列
        result = merge_ordered(left, right, on="E", left_by=["G", "H"])
        # 创建预期的 DataFrame 结果
        expected = DataFrame(
            {"G": ["g"] * 3, "H": ["h"] * 3, "E": [1, 2, 3], "T": [np.nan, 1.0, np.nan]}
        )

        # 使用测试工具比较 result 和 expected 的数据框是否相等
        tm.assert_frame_equal(result, expected)

    def test_elements_not_in_by_but_in_df(self):
        # GH 38167
        # 创建左侧 DataFrame，包含两列数据
        left = DataFrame([["g", "h", 1], ["g", "h", 3]], columns=list("GHE"))
        # 创建右侧 DataFrame，包含两列数据
        right = DataFrame([[2, 1]], columns=list("ET"))
        # 设置错误消息的正则表达式
        msg = r"\{'h'\} not found in left columns"
        # 使用 pytest 的 raises 断言，检查是否会抛出 KeyError，并匹配错误消息
        with pytest.raises(KeyError, match=msg):
            # 调用 merge_ordered 函数，尝试使用列名 "h" 作为 left_by 参数进行合并
            merge_ordered(left, right, on="E", left_by=["G", "h"])

    @pytest.mark.parametrize("invalid_method", ["linear", "carrot"])
    def test_ffill_validate_fill_method(self, left, right, invalid_method):
        # GH 55884
        # 使用 pytest 的 raises 断言，检查是否会抛出 ValueError，且错误消息中包含 "fill_method must be 'ffill' or None"
        with pytest.raises(
            ValueError, match=re.escape("fill_method must be 'ffill' or None")
        ):
            # 调用 merge_ordered 函数，尝试使用无效的 fill_method 参数值进行合并
            merge_ordered(left, right, on="key", fill_method=invalid_method)

    def test_ffill_left_merge(self):
        # GH 57010
        # 创建第一个 DataFrame，包含三列数据
        df1 = DataFrame(
            {
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3, 1, 2, 3],
                "group": ["a", "a", "a", "b", "b", "b"],
            }
        )
        # 创建第二个 DataFrame，包含两列数据
        df2 = DataFrame({"key": ["b", "c", "d"], "rvalue": [1, 2, 3]})
        # 使用 merge_ordered 函数，以 group 列为左侧合并的关键列，使用前向填充方法 ("ffill")，左连接方式进行合并
        result = merge_ordered(
            df1, df2, fill_method="ffill", left_by="group", how="left"
        )
        # 创建预期的 DataFrame 结果
        expected = DataFrame(
            {
                "key": ["a", "c", "e", "a", "c", "e"],
                "lvalue": [1, 2, 3, 1, 2, 3],
                "group": ["a", "a", "a", "b", "b", "b"],
                "rvalue": [np.nan, 2.0, 2.0, np.nan, 2.0, 2.0],
            }
        )
        # 使用测试工具比较 result 和 expected 的数据框是否相等
        tm.assert_frame_equal(result, expected)
```