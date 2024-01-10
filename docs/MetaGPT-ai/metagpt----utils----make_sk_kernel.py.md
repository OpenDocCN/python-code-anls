# `MetaGPT\metagpt\utils\make_sk_kernel.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/9/13 12:29
@Author  : femto Zheng
@File    : make_sk_kernel.py
"""
# 导入所需的模块和类
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import (
    OpenAIChatCompletion,
)

from metagpt.config import CONFIG

# 定义一个函数，用于创建语义内核
def make_sk_kernel():
    # 创建一个语义内核对象
    kernel = sk.Kernel()
    # 根据配置选择使用 Azure 还是 OpenAI 的聊天完成服务
    if CONFIG.OPENAI_API_TYPE == "azure":
        # 如果配置为 Azure，添加 Azure 聊天完成服务
        kernel.add_chat_service(
            "chat_completion",
            AzureChatCompletion(CONFIG.DEPLOYMENT_NAME, CONFIG.OPENAI_BASE_URL, CONFIG.OPENAI_API_KEY),
        )
    else:
        # 如果配置为其他，添加 OpenAI 聊天完成服务
        kernel.add_chat_service(
            "chat_completion",
            OpenAIChatCompletion(CONFIG.OPENAI_API_MODEL, CONFIG.OPENAI_API_KEY),
        )

    # 返回创建好的语义内核对象
    return kernel

```