# `MetaGPT\tests\metagpt\provider\test_azure_openai_api.py`

```py

#!/usr/bin/env python
# 指定使用 Python 解释器来执行脚本

# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8

# @Desc   :
# 代码描述部分，可能用于说明代码的作用或者功能

# 导入所需的模块
from metagpt.config import CONFIG
from metagpt.provider.azure_openai_api import AzureOpenAILLM

# 设置 OpenAI API 的版本为 "xx"
CONFIG.OPENAI_API_VERSION = "xx"

# 设置 OpenAI API 的代理为 "http://127.0.0.1:80"，这里是一个假值
CONFIG.openai_proxy = "http://127.0.0.1:80"  # fake value

# 定义测试函数 test_azure_openai_api
def test_azure_openai_api():
    # 创建 AzureOpenAILLM 的实例，但是没有赋值给任何变量
    _ = AzureOpenAILLM()

```