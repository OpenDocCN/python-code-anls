# `MetaGPT\tests\metagpt\utils\test_output_parser.py`

```

#!/usr/bin/env python
# coding: utf-8
"""
@Time    : 2023/7/11 10:25
@Author  : chengmaoyu
@File    : test_output_parser.py
"""
# 导入所需的模块
from typing import List, Tuple, Union
import pytest
from metagpt.utils.common import OutputParser

# 测试函数：测试解析文本中的块
def test_parse_blocks():
    test_text = "##block1\nThis is block 1.\n##block2\nThis is block 2."
    expected_result = {"block1": "This is block 1.", "block2": "This is block 2."}
    assert OutputParser.parse_blocks(test_text) == expected_result

# 测试函数：测试解析代码块
def test_parse_code():
    test_text = "```python\nprint('Hello, world!')```"
    expected_result = "print('Hello, world!')"
    assert OutputParser.parse_code(test_text, "python") == expected_result
    # 测试异常情况
    with pytest.raises(Exception):
        OutputParser.parse_code(test_text, "java")

# 测试函数：测试解析 Python 代码块
def test_parse_python_code():
    expected_result = "print('Hello, world!')"
    assert OutputParser.parse_python_code("```python\nprint('Hello, world!')```") == expected_result
    # ...（其他测试用例）
    with pytest.raises(ValueError):
        OutputParser.parse_python_code("xxx =")

# 测试函数：测试解析字符串
def test_parse_str():
    test_text = "name = 'Alice'"
    expected_result = "Alice"
    assert OutputParser.parse_str(test_text) == expected_result

# 测试函数：测试解析文件列表
def test_parse_file_list():
    test_text = "files=['file1', 'file2', 'file3']"
    expected_result = ["file1", "file2", "file3"]
    assert OutputParser.parse_file_list(test_text) == expected_result
    # 测试异常情况
    # with pytest.raises(Exception):
    #     OutputParser.parse_file_list("wrong_input")

# 测试函数：测试解析数据
def test_parse_data():
    test_data = "##block1\n```python\nprint('Hello, world!')\n```\n##block2\nfiles=['file1', 'file2', 'file3']"
    expected_result = {"block1": "print('Hello, world!')\n", "block2": ["file1", "file2", "file3"]}
    assert OutputParser.parse_data(test_data) == expected_result

# 测试函数：测试提取结构
@pytest.mark.parametrize(
    ("text", "data_type", "parsed_data", "expected_exception"),
    [
        # ...（其他测试用例）
    ],
)
def test_extract_struct(
    text: str, data_type: Union[type(list), type(dict)], parsed_data: Union[list, dict], expected_exception
):
    def case():
        resp = OutputParser.extract_struct(text, data_type)
        assert resp == parsed_data

    if expected_exception:
        with pytest.raises(expected_exception):
            case()
    else:
        case()

# 测试函数：测试带有 Markdown 映射的解析
def test_parse_with_markdown_mapping():
    OUTPUT_MAPPING = {
        "Original Requirements": (str, ...),
        "Product Goals": (List[str], ...),
        "User Stories": (List[str], ...),
        "Competitive Analysis": (List[str], ...),
        "Competitive Quadrant Chart": (str, ...),
        "Requirement Analysis": (str, ...),
        "Requirement Pool": (List[Tuple[str, str]], ...),
        "Anything UNCLEAR": (str, ...),
    }
    t_text_with_content_tag = """[CONTENT]## Original Requirements: ... [/CONTENT]"""
    t_text_raw = t_text_with_content_tag.replace("[CONTENT]", "").replace("[/CONTENT]", "")
    d = OutputParser.parse_data_with_mapping(t_text_with_content_tag, OUTPUT_MAPPING)
    # 打印结果
    import json
    print(json.dumps(d))
    assert d["Original Requirements"] == t_text_raw.split("## Original Requirements:")[1].split("##")[0].strip()

```