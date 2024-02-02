# `MetaGPT\tests\metagpt\provider\postprocess\test_base_postprocess_plugin.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 说明文件的编码格式和描述信息

from metagpt.provider.postprocess.base_postprocess_plugin import BasePostProcessPlugin
# 导入基础后处理插件类

raw_output = """
[CONTENT]
{
"Original Requirements": "xxx"
}
[/CONTENT]
"""
# 原始输出数据，包含原始需求信息的 JSON 格式字符串

raw_schema = {
    "title": "prd",
    "type": "object",
    "properties": {
        "Original Requirements": {"title": "Original Requirements", "type": "string"},
    },
    "required": [
        "Original Requirements",
    ],
}
# 原始数据的模式，用于验证和处理输出数据的结构

def test_llm_post_process_plugin():
    # 创建基础后处理插件对象
    post_process_plugin = BasePostProcessPlugin()

    # 运行后处理插件，处理原始输出数据并返回处理后的结果
    output = post_process_plugin.run(output=raw_output, schema=raw_schema)
    # 断言处理后的结果中包含"Original Requirements"字段
    assert "Original Requirements" in output

```