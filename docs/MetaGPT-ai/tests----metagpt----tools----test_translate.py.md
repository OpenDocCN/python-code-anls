# `MetaGPT\tests\metagpt\tools\test_translate.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/2 17:46
@Author  : alexanderwu
@File    : test_translate.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger
# 从 metagpt.tools.translator 模块中导入 Translator 类
from metagpt.tools.translator import Translator

# 标记该测试函数为异步函数
@pytest.mark.asyncio
# 使用 llm_api 作为测试函数的 fixture
@pytest.mark.usefixtures("llm_api")
# 定义测试函数 test_translate，参数为 llm_api
async def test_translate(llm_api):
    # 定义两个测试用例，每个测试用例包含一个英文句子和对应的翻译结果
    poetries = [
        ("Let life be beautiful like summer flowers", "花"),
        ("The ancient Chinese poetries are all songs.", "中国"),
    ]
    # 遍历测试用例
    for i, j in poetries:
        # 调用 Translator 类的 translate_prompt 方法，将英文句子转换为 GPT-3 的输入格式
        prompt = Translator.translate_prompt(i)
        # 调用 llm_api 的 aask 方法，向 GPT-3 提出翻译请求
        rsp = await llm_api.aask(prompt)
        # 记录翻译结果
        logger.info(rsp)
        # 断言翻译结果中包含预期的翻译文本
        assert j in rsp

```