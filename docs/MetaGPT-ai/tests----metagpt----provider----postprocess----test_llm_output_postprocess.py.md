# `MetaGPT\tests\metagpt\provider\postprocess\test_llm_output_postprocess.py`

```py

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定编码格式为 UTF-8
# @Desc   : 
# 描述信息为空

# 从metagpt.provider.postprocess.llm_output_postprocess模块中导入llm_output_postprocess函数
from metagpt.provider.postprocess.llm_output_postprocess import llm_output_postprocess
# 从tests.metagpt.provider.postprocess.test_base_postprocess_plugin模块中导入raw_output和raw_schema变量
from tests.metagpt.provider.postprocess.test_base_postprocess_plugin import (
    raw_output,
    raw_schema,
)

# 定义测试函数test_llm_output_postprocess
def test_llm_output_postprocess():
    # 调用llm_output_postprocess函数，传入raw_output和raw_schema作为参数
    output = llm_output_postprocess(output=raw_output, schema=raw_schema)
    # 断言"Original Requirements"在输出中
    assert "Original Requirements" in output

```