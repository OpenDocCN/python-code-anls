# `MetaGPT\examples\write_tutorial.py`

```py

#!/usr/bin/env python3
# _*_ coding: utf-8 _*_

"""
@Time    : 2023/9/4 21:40:57
@Author  : Stitch-z
@File    : tutorial_assistant.py
"""

# 导入 asyncio 模块
import asyncio

# 从 metagpt.roles.tutorial_assistant 模块中导入 TutorialAssistant 类
from metagpt.roles.tutorial_assistant import TutorialAssistant

# 定义异步函数 main
async def main():
    # 定义主题
    topic = "Write a tutorial about MySQL"
    # 创建 TutorialAssistant 角色对象，指定语言为中文
    role = TutorialAssistant(language="Chinese")
    # 运行角色的任务
    await role.run(topic)

# 如果当前脚本为主程序
if __name__ == "__main__":
    # 运行异步函数 main
    asyncio.run(main())

```