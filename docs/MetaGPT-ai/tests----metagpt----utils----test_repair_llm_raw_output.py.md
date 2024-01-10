# `MetaGPT\tests\metagpt\utils\test_repair_llm_raw_output.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : unittest of repair_llm_raw_output

from metagpt.config import CONFIG

"""
CONFIG.repair_llm_output should be True before retry_parse_json_text imported.
so we move `from ... impot ...` into each `test_xx` to avoid `Module level import not at top of file` format warning.
"""
CONFIG.repair_llm_output = True

# 测试修复大小写敏感问题
def test_repair_case_sensitivity():
    from metagpt.utils.repair_llm_raw_output import repair_llm_raw_output

    # 原始输出
    raw_output = """{
    "Original requirements": "Write a 2048 game",
    "search Information": "",
    "competitive Quadrant charT": "quadrantChart
                Campaign A: [0.3, 0.6]",
    "requirement analysis": "The 2048 game should be simple to play"
}"""
    # 修复后的目标输出
    target_output = """{
    "Original Requirements": "Write a 2048 game",
    "Search Information": "",
    "Competitive Quadrant Chart": "quadrantChart
                Campaign A: [0.3, 0.6]",
    "Requirement Analysis": "The 2048 game should be simple to play"
}"""
    # 修复大小写敏感问题
    req_keys = ["Original Requirements", "Search Information", "Competitive Quadrant Chart", "Requirement Analysis"]
    output = repair_llm_raw_output(output=raw_output, req_keys=req_keys)
    assert output == target_output

# 测试修复特殊字符缺失问题
def test_repair_special_character_missing():
    from metagpt.utils.repair_llm_raw_output import repair_llm_raw_output

    # 原始输出
    raw_output = """[CONTENT]
    "Anything UNCLEAR": "No unclear requirements or information."
[CONTENT]"""
    # 修复后的目标输出
    target_output = """[CONTENT]
    "Anything UNCLEAR": "No unclear requirements or information."
[/CONTENT]"""
    # 修复特殊字符缺失问题
    req_keys = ["[/CONTENT]"]
    output = repair_llm_raw_output(output=raw_output, req_keys=req_keys)
    assert output == target_output

    # 其他测试用例略

# 测试修复必需键值对缺失问题
def test_required_key_pair_missing():
    from metagpt.utils.repair_llm_raw_output import repair_llm_raw_output

    # 其他测试用例略

# 测试修复 JSON 格式问题
def test_repair_json_format():
    from metagpt.utils.repair_llm_raw_output import RepairType, repair_llm_raw_output

    # 其他测试用例略

# 测试修复无效 JSON 问题
def test_repair_invalid_json():
    from metagpt.utils.repair_llm_raw_output import repair_invalid_json

    # 其他测试用例略

# 测试重试解析 JSON 文本
def test_retry_parse_json_text():
    from metagpt.utils.repair_llm_raw_output import retry_parse_json_text

    # 其他测试用例略

# 提取输出内容
def extract_content_from_output(output):
    # 提取输出内容
    output = extract_content_from_output(output)
    assert output.startswith('{\n"Implementation approach"') and output.endswith(
        '"Anything UNCLEAR": "The requirement is clear to me."\n}'
    )

```