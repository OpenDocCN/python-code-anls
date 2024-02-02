# `MetaGPT\examples\research.py`

```py

#!/usr/bin/env python
# 指定使用 python 解释器来执行脚本

import asyncio
# 导入异步 I/O 模块

from metagpt.roles.researcher import RESEARCH_PATH, Researcher
# 从 metagpt.roles.researcher 模块中导入 RESEARCH_PATH 和 Researcher 类

async def main():
    # 定义异步函数 main
    topic = "dataiku vs. datarobot"
    # 定义主题字符串
    role = Researcher(language="en-us")
    # 创建 Researcher 对象，指定语言为 en-us
    await role.run(topic)
    # 调用 Researcher 对象的 run 方法，传入主题字符串
    print(f"save report to {RESEARCH_PATH / f'{topic}.md'}.")
    # 打印保存报告的路径

if __name__ == "__main__":
    # 如果当前脚本作为主程序执行
    asyncio.run(main())
    # 运行异步函数 main

```