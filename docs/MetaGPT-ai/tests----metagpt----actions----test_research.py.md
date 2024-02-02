# `MetaGPT\tests\metagpt\actions\test_research.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/12/28
@Author  : mashenquan
@File    : test_research.py
"""

# 导入 pytest 模块
import pytest

# 从 metagpt.actions 模块中导入 research 模块
from metagpt.actions import research

# 标记异步测试
@pytest.mark.asyncio
async def test_collect_links(mocker):
    # 定义模拟的 LLN 提问函数
    async def mock_llm_ask(self, prompt: str, system_msgs):
        # 如果提示中包含关键字，则返回指定的关键字列表
        if "Please provide up to 2 necessary keywords" in prompt:
            return '["metagpt", "llm"]'
        # 如果提示中包含查询相关主题的要求，则返回指定的查询列表
        elif "Provide up to 4 queries related to your research topic" in prompt:
            return (
                '["MetaGPT use cases", "The roadmap of MetaGPT", '
                '"The function of MetaGPT", "What llm MetaGPT support"]'
            )
        # 如果提示中包含排序搜索结果的要求，则返回指定的排序结果
        elif "sort the remaining search results" in prompt:
            return "[1,2]"

    # 使用 mocker 模块的 patch 方法模拟 LLN 提问函数
    mocker.patch("metagpt.provider.base_llm.BaseLLM.aask", mock_llm_ask)
    # 运行 CollectLinks 类的 run 方法，传入指定的参数
    resp = await research.CollectLinks().run("The application of MetaGPT")
    # 断言结果中是否包含指定的查询列表
    for i in ["MetaGPT use cases", "The roadmap of MetaGPT", "The function of MetaGPT", "What llm MetaGPT support"]:
        assert i in resp

# ...（以下部分代码注释与上述类似，不再赘述）

```