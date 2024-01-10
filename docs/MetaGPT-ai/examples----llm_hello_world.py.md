# `MetaGPT\examples\llm_hello_world.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/6 14:13
@Author  : alexanderwu
@File    : llm_hello_world.py
"""
# 导入 asyncio 模块
import asyncio
# 从 metagpt.llm 模块中导入 LLM 类
from metagpt.llm import LLM
# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger

# 定义异步函数 main
async def main():
    # 创建 LLM 对象
    llm = LLM()
    # 记录日志并等待 LLM 对象的 aask 方法返回结果
    logger.info(await llm.aask("hello world"))
    # 记录日志并等待 LLM 对象的 aask_batch 方法返回结果
    logger.info(await llm.aask_batch(["hi", "write python hello world."]))

    # 定义 hello_msg 变量
    hello_msg = [{"role": "user", "content": "count from 1 to 10. split by newline."}]
    # 记录日志并等待 LLM 对象的 acompletion 方法返回结果
    logger.info(await llm.acompletion(hello_msg))
    # 记录日志并等待 LLM 对象的 acompletion_text 方法返回结果
    logger.info(await llm.acompletion_text(hello_msg))

    # 流式模式，速度较慢
    # 等待 LLM 对象的 acompletion_text 方法以流式模式返回结果
    await llm.acompletion_text(hello_msg, stream=True)

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行异步函数 main
    asyncio.run(main())

```