# `.\DB-GPT-src\dbgpt\client\app.py`

```py
"""App Client API."""

# 引入所需模块和类型提示
from typing import List

from dbgpt.core.schema.api import Result

# 导入本地模块和异常处理类
from .client import Client, ClientException
from .schema import AppModel


async def get_app(client: Client, app_id: str) -> AppModel:
    """获取单个应用信息。

    Args:
        client (Client): dbgpt 客户端对象。
        app_id (str): 应用程序的 ID。
    Returns:
        AppModel: 应用程序模型对象。
    Raises:
        ClientException: 如果请求失败。
    """
    try:
        # 使用客户端对象发起 GET 请求，获取应用信息
        res = await client.get("/apps/" + app_id)
        # 将响应内容解析为 Result 对象
        result: Result = res.json()
        if result["success"]:
            # 如果请求成功，返回从数据中构建的 AppModel 对象
            return AppModel(**result["data"])
        else:
            # 如果请求失败，抛出 ClientException 异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，抛出 ClientException 异常，并将异常信息作为原因
        raise ClientException(f"Failed to get app: {e}")


async def list_app(client: Client) -> List[AppModel]:
    """列出所有应用程序。

    Args:
        client (Client): dbgpt 客户端对象。
    Returns:
        List[AppModel]: 应用程序模型对象的列表。
    Raises:
        ClientException: 如果请求失败。
    """
    try:
        # 使用客户端对象发起 GET 请求，获取应用列表信息
        res = await client.get("/apps")
        # 将响应内容解析为 Result 对象
        result: Result = res.json()
        if result["success"]:
            # 如果请求成功，返回从数据中构建的 AppModel 对象列表
            return [AppModel(**app) for app in result["data"]["app_list"]]
        else:
            # 如果请求失败，抛出 ClientException 异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，抛出 ClientException 异常，并将异常信息作为原因
        raise ClientException(f"Failed to list apps: {e}")
```