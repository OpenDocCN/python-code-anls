# `MetaGPT\metagpt\provider\anthropic_api.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/21 11:15
@Author  : Leo Xiao
@File    : anthropic_api.py
"""

# 导入 anthropic 模块
import anthropic
# 从 anthropic 模块中导入 Anthropic 和 AsyncAnthropic 类
from anthropic import Anthropic, AsyncAnthropic
# 从 metagpt.config 模块中导入 CONFIG 变量
from metagpt.config import CONFIG

# 定义 Claude2 类
class Claude2:
    # 定义 ask 方法，接受一个字符串类型的参数 prompt，返回一个字符串类型的结果
    def ask(self, prompt: str) -> str:
        # 创建 Anthropic 客户端对象，使用 CONFIG.anthropic_api_key 作为 API 密钥
        client = Anthropic(api_key=CONFIG.anthropic_api_key)

        # 调用客户端对象的 completions.create 方法，传入相关参数，获取结果
        res = client.completions.create(
            model="claude-2",
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=1000,
        )
        # 返回结果的 completion 属性
        return res.completion

    # 定义 aask 方法，接受一个字符串类型的参数 prompt，返回一个字符串类型的结果
    async def aask(self, prompt: str) -> str:
        # 创建 AsyncAnthropic 客户端对象，使用 CONFIG.anthropic_api_key 作为 API 密钥
        aclient = AsyncAnthropic(api_key=CONFIG.anthropic_api_key)

        # 调用客户端对象的 completions.create 方法，传入相关参数，获取结果
        res = await aclient.completions.create(
            model="claude-2",
            prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=1000,
        )
        # 返回结果的 completion 属性
        return res.completion

```