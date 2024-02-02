# `MetaGPT\examples\search_with_specific_engine.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
# 导入 asyncio 模块
import asyncio

# 从 metagpt.roles 模块中导入 Searcher 类
from metagpt.roles import Searcher
# 从 metagpt.tools 模块中导入 SearchEngineType 枚举类型
from metagpt.tools import SearchEngineType

# 定义异步函数 main
async def main():
    # 设置问题
    question = "What are the most interesting human facts?"
    # 使用 SERPAPI_GOOGLE 引擎进行搜索并运行
    await Searcher(engine=SearchEngineType.SERPAPI_GOOGLE).run(question)

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行异步函数 main
    asyncio.run(main())

```