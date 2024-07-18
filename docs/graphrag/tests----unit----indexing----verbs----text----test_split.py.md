# `.\graphrag\tests\unit\indexing\verbs\text\test_split.py`

```py
# 导入单元测试相关的模块
import unittest

# 导入 pandas 库并简称为 pd
import pandas as pd

# 导入 pytest 库用于测试异常情况
import pytest

# 从指定路径导入 text_split_df 函数
from graphrag.index.verbs.text.split import text_split_df

# 定义 TestTextSplit 类，继承自 unittest.TestCase，用于测试 text_split_df 函数
class TestTextSplit(unittest.TestCase):

    # 测试空字符串的情况
    def test_empty_string(self):
        # 创建包含一个空字符串的 DataFrame
        input = pd.DataFrame([{"in": ""}])
        # 调用 text_split_df 函数，将结果转换为字典列表
        result = text_split_df(input, "in", "out", ",").to_dict(orient="records")

        # 断言结果长度为1
        assert len(result) == 1
        # 断言输出列表为空
        assert result[0]["out"] == []

    # 测试没有分隔符的字符串情况
    def test_string_without_seperator(self):
        # 创建包含一个没有分隔符的字符串的 DataFrame
        input = pd.DataFrame([{"in": "test_string_without_seperator"}])
        # 调用 text_split_df 函数，将结果转换为字典列表
        result = text_split_df(input, "in", "out", ",").to_dict(orient="records")

        # 断言结果长度为1
        assert len(result) == 1
        # 断言输出列表包含原始字符串作为单个元素
        assert result[0]["out"] == ["test_string_without_seperator"]

    # 测试带分隔符的字符串情况
    def test_string_with_seperator(self):
        # 创建包含一个带分隔符的字符串的 DataFrame
        input = pd.DataFrame([{"in": "test_1,test_2"}])
        # 调用 text_split_df 函数，将结果转换为字典列表
        result = text_split_df(input, "in", "out", ",").to_dict(orient="records")

        # 断言结果长度为1
        assert len(result) == 1
        # 断言输出列表包含分隔后的多个字符串
        assert result[0]["out"] == ["test_1", "test_2"]

    # 测试包含列表作为列值的情况
    def test_row_with_list_as_column(self):
        # 创建包含一个列表作为列值的 DataFrame
        input = pd.DataFrame([{"in": ["test_1", "test_2"]}])
        # 调用 text_split_df 函数，将结果转换为字典列表
        result = text_split_df(input, "in", "out", ",").to_dict(orient="records")

        # 断言结果长度为1
        assert len(result) == 1
        # 断言输出列表与原始列表相同
        assert result[0]["out"] == ["test_1", "test_2"]

    # 测试非字符串列值会抛出异常的情况
    def test_non_string_column_throws_error(self):
        # 创建包含一个非字符串列值的 DataFrame
        input = pd.DataFrame([{"in": 5}])
        # 使用 pytest 的断言检测是否抛出 TypeError 异常
        with pytest.raises(TypeError):
            text_split_df(input, "in", "out", ",").to_dict(orient="records")

    # 测试多行数据返回正确的情况
    def test_more_than_one_row_returns_correctly(self):
        # 创建包含多行数据的 DataFrame
        input = pd.DataFrame([{"in": "row_1_1,row_1_2"}, {"in": "row_2_1,row_2_2"}])
        # 调用 text_split_df 函数，将结果转换为字典列表
        result = text_split_df(input, "in", "out", ",").to_dict(orient="records")

        # 断言结果长度为2
        assert len(result) == 2
        # 断言每行的输出列表与预期的字符串分隔结果相符
        assert result[0]["out"] == ["row_1_1", "row_1_2"]
        assert result[1]["out"] == ["row_2_1", "row_2_2"]
```