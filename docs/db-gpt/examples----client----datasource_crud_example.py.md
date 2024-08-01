# `.\DB-GPT-src\examples\client\datasource_crud_example.py`

```py
"""Client: Simple Flow CRUD example

This example demonstrates how to use the dbgpt client to create, get, update, and
delete datasource.

Example:
    .. code-block:: python

        DBGPT_API_KEY = "dbgpt"
        client = Client(api_key=DBGPT_API_KEY)
        # 1. Create a flow
        res = await create_datasource(
            client,
            DatasourceModel(
                db_name="dbgpt",
                desc="for client datasource",
                db_type="mysql",
                db_host="127.0.0.1",
                db_user="root",
                db_pwd="xxxx",
                db_port=3306,
            ),
        )
        # 2. Update a flow
        res = await update_datasource(
            client,
            DatasourceModel(
                db_name="dbgpt",
                desc="for client datasource",
                db_type="mysql",
                db_host="127.0.0.1",
                db_user="root",
                db_pwd="xxxx",
                db_port=3306,
            ),
        )
        # 3. Delete a flow
        res = await delete_datasource(client, datasource_id="10")
        # 4. Get a flow
        res = await get_datasource(client, datasource_id="10")
        # 5. List all datasource
        res = await list_datasource(client)

"""
import asyncio

from dbgpt.client import Client
from dbgpt.client.datasource import list_datasource


async def main():
    # 初始化客户端并设置 API 密钥
    DBGPT_API_KEY = "dbgpt"
    client = Client(api_key=DBGPT_API_KEY)
    
    # 调用 list_datasource 方法获取所有数据源的信息
    res = await list_datasource(client)
    
    # 打印获取的结果
    print(res)


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())
```