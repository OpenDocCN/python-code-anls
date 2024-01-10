# `MetaGPT\examples\search_google.py`

```

#!/usr/bin/env python
# 指定解释器为 python
# -*- coding: utf-8 -*-
# 指定文件编码格式为 UTF-8
"""
@Time    : 2023/5/7 18:32
@Author  : alexanderwu
@File    : search_google.py
"""
# 文件的注释信息，包括时间、作者和文件名

import asyncio
# 导入 asyncio 模块

from metagpt.roles import Searcher
# 从 metagpt.roles 模块中导入 Searcher 类

async def main():
    # 定义异步函数 main
    await Searcher().run("What are some good sun protection products?")
    # 调用 Searcher 类的 run 方法并传入参数

if __name__ == "__main__":
    # 如果当前文件被直接执行
    asyncio.run(main())
    # 运行异步函数 main

```