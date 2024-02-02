# `MetaGPT\tests\metagpt\provider\test_metagpt_api.py`

```py

#!/usr/bin/env python
# 指定解释器为 Python
# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8
"""
@Time    : 2023/12/28
@Author  : mashenquan
@File    : test_metagpt_api.py
"""
# 文件的时间、作者和名称信息
from metagpt.config import LLMProviderEnum
# 从 metagpt.config 模块中导入 LLMProviderEnum 类
from metagpt.llm import LLM
# 从 metagpt.llm 模块中导入 LLM 类

def test_llm():
    # 创建 LLM 对象，指定提供者为 METAGPT
    llm = LLM(provider=LLMProviderEnum.METAGPT)
    # 断言 LLM 对象存在
    assert llm

```