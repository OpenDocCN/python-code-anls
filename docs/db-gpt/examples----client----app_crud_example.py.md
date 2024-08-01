# `.\DB-GPT-src\examples\client\app_crud_example.py`

```py
"""Client: Simple App CRUD example.

This example demonstrates how to use the dbgpt client to get, list apps.
Example:
    .. code-block:: python

        DBGPT_API_KEY = "dbgpt"
        client = Client(api_key=DBGPT_API_KEY)
        # 1. List all apps
        res = await list_app(client)
        # 2. Get an app
        res = await get_app(client, app_id="bf1c7561-13fc-4fe0-bf5d-c22e724766a8")
"""
import asyncio  # 导入异步I/O模块

from dbgpt.client import Client  # 从dbgpt.client模块导入Client类
from dbgpt.client.app import list_app  # 从dbgpt.client.app模块导入list_app函数


async def main():
    # initialize client
    DBGPT_API_KEY = "dbgpt"  # 设定调试工具API密钥
    client = Client(api_key=DBGPT_API_KEY)  # 创建一个使用指定API密钥的Client对象
    res = await list_app(client)  # 调用list_app函数，传入client对象，返回应用列表
    print(res)  # 打印返回结果


if __name__ == "__main__":
    asyncio.run(main())  # 运行main函数，使用asyncio.run来运行异步函数
```