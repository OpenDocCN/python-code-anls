# `MetaGPT\examples\search_kb.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File    : search_kb.py
@Modified By: mashenquan, 2023-12-22. Delete useless codes.
"""
# 导入 asyncio 模块
import asyncio

# 从 langchain.embeddings 模块中导入 OpenAIEmbeddings 类
from langchain.embeddings import OpenAIEmbeddings

# 从 metagpt.config 模块中导入 CONFIG 变量
from metagpt.config import CONFIG
# 从 metagpt.const 模块中导入 DATA_PATH 和 EXAMPLE_PATH 变量
from metagpt.const import DATA_PATH, EXAMPLE_PATH
# 从 metagpt.document_store 模块中导入 FaissStore 类
from metagpt.document_store import FaissStore
# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger
# 从 metagpt.roles 模块中导入 Sales 类
from metagpt.roles import Sales

# 定义函数 get_store
def get_store():
    # 创建 OpenAIEmbeddings 对象
    embedding = OpenAIEmbeddings(openai_api_key=CONFIG.openai_api_key, openai_api_base=CONFIG.openai_base_url)
    # 返回 FaissStore 对象
    return FaissStore(DATA_PATH / "example.json", embedding=embedding)

# 定义异步函数 search
async def search():
    # 创建 FaissStore 对象
    store = FaissStore(EXAMPLE_PATH / "example.json")
    # 创建 Sales 角色对象
    role = Sales(profile="Sales", store=store)
    # 定义查询字符串
    query = "Which facial cleanser is good for oily skin?"
    # 运行角色的查询方法，获取结果
    result = await role.run(query)
    # 记录结果
    logger.info(result)

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行异步函数 search
    asyncio.run(search())

```