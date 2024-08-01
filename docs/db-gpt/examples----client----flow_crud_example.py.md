# `.\DB-GPT-src\examples\client\flow_crud_example.py`

```py
"""Client: Simple Flow CRUD example

This example demonstrates how to use the dbgpt client to create, get, update, and
delete flows.

Example:
    .. code-block:: python

        DBGPT_API_KEY = "dbgpt"
        client = Client(api_key=DBGPT_API_KEY)
        # 1. Create a flow
        res = await create_flow(
            client,
            FlowPanel(name="test_flow", desc="for client flow", owner="dbgpt"),
        )
        # 2. Update a flow
        res = await update_flow(
            client,
            FlowPanel(name="test_flow", desc="for client flow333", owner="dbgpt"),
        )
        # 3. Delete a flow
        res = await delete_flow(client, flow_id="bf1c7561-13fc-4fe0-bf5d-c22e724766a8")
        # 4. Get a flow
        res = await get_flow(client, flow_id="bf1c7561-13fc-4fe0-bf5d-c22e724766a8")
        # 5. List all flows
        res = await list_flow(client)

"""
import asyncio

from dbgpt.client import Client
from dbgpt.client.flow import list_flow

# 引入异步I/O操作模块

async def main():
    # 初始化客户端，使用指定的API密钥创建客户端对象
    DBGPT_API_KEY = "dbgpt"
    client = Client(api_key=DBGPT_API_KEY)
    # 调用列表流程的函数，获取所有流程的信息
    res = await list_flow(client)
    # 打印返回的结果
    print(res)


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())
```