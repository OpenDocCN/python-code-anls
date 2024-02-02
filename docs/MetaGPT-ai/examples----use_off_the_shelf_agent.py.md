# `MetaGPT\examples\use_off_the_shelf_agent.py`

```py
"""
Filename: MetaGPT/examples/use_off_the_shelf_agent.py
Created Date: Tuesday, September 19th 2023, 6:52:25 pm
Author: garylin2099
"""
# 导入必要的模块
import asyncio
# 导入日志模块
from metagpt.logs import logger
# 导入产品经理角色模块
from metagpt.roles.product_manager import ProductManager

# 异步函数，用于执行主程序
async def main():
    # 定义消息内容
    msg = "Write a PRD for a snake game"
    # 创建产品经理角色对象
    role = ProductManager()
    # 执行产品经理角色的任务，并获取结果
    result = await role.run(msg)
    # 记录结果的前100个字符
    logger.info(result.content[:100])

# 如果当前文件被直接执行
if __name__ == "__main__":
    # 运行主程序
    asyncio.run(main())
"""
```