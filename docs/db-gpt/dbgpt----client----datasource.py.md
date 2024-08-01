# `.\DB-GPT-src\dbgpt\client\datasource.py`

```py
"""this module contains the datasource client functions."""
# 导入所需模块和函数

from typing import List  # 引入 List 类型提示

from dbgpt._private.pydantic import model_to_dict  # 导入模型转换函数
from dbgpt.core.schema.api import Result  # 导入结果模型

from .client import Client, ClientException  # 导入客户端和客户端异常类
from .schema import DatasourceModel  # 导入数据源模型


async def create_datasource(
    client: Client, datasource: DatasourceModel
) -> DatasourceModel:
    """Create a new datasource.

    Args:
        client (Client): The dbgpt client.
        datasource (DatasourceModel): The datasource model.
    """
    try:
        # 发起 POST 请求创建数据源，使用数据源模型的字典表示
        res = await client.get("/datasources", model_to_dict(datasource))
        result: Result = res.json()  # 解析返回的 JSON 结果
        if result["success"]:
            # 如果请求成功，返回解析后的数据源模型
            return DatasourceModel(**result["data"])
        else:
            # 如果请求失败，抛出客户端异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常并转换为客户端异常
        raise ClientException(f"Failed to create datasource: {e}")


async def update_datasource(
    client: Client, datasource: DatasourceModel
) -> DatasourceModel:
    """Update a datasource.

    Args:
        client (Client): The dbgpt client.
        datasource (DatasourceModel): The datasource model.
    Returns:
        DatasourceModel: The datasource model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起 PUT 请求更新数据源，使用数据源模型的字典表示
        res = await client.put("/datasources", model_to_dict(datasource))
        result: Result = res.json()  # 解析返回的 JSON 结果
        if result["success"]:
            # 如果请求成功，返回解析后的数据源模型
            return DatasourceModel(**result["data"])
        else:
            # 如果请求失败，抛出客户端异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常并转换为客户端异常
        raise ClientException(f"Failed to update datasource: {e}")


async def delete_datasource(client: Client, datasource_id: str) -> DatasourceModel:
    """
    Delete a datasource.

    Args:
        client (Client): The dbgpt client.
        datasource_id (str): The datasource id.
    Returns:
        DatasourceModel: The datasource model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起 DELETE 请求删除数据源
        res = await client.delete("/datasources/" + datasource_id)
        result: Result = res.json()  # 解析返回的 JSON 结果
        if result["success"]:
            # 如果请求成功，返回解析后的数据源模型
            return DatasourceModel(**result["data"])
        else:
            # 如果请求失败，抛出客户端异常
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常并转换为客户端异常
        raise ClientException(f"Failed to delete datasource: {e}")


async def get_datasource(client: Client, datasource_id: str) -> DatasourceModel:
    """
    Get a datasource.

    Args:
        client (Client): The dbgpt client.
        datasource_id (str): The datasource id.
    Returns:
        DatasourceModel: The datasource model.
    Raises:
        ClientException: If the request failed.
    """
    # 尝试通过客户端从指定路径获取数据源信息，其中datasource_id是路径的一部分
    try:
        # 使用异步方式发送GET请求，获取响应对象res
        res = await client.get("/datasources/" + datasource_id)
        # 将响应的JSON内容解析为Result对象
        result: Result = res.json()
        # 检查返回结果中的success字段，确定是否成功获取数据源信息
        if result["success"]:
            # 如果成功，将返回的数据转换为DatasourceModel对象并返回
            return DatasourceModel(**result["data"])
        else:
            # 如果未成功，抛出客户端异常，包括错误码和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，并将其作为原因抛出客户端异常，提示获取数据源失败
        raise ClientException(f"Failed to get datasource: {e}")
# 异步函数，用于列出数据源
async def list_datasource(client: Client) -> List[DatasourceModel]:
    """
    List datasources.

    Args:
        client (Client): The dbgpt client.
    Returns:
        List[DatasourceModel]: The list of datasource models.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发送 GET 请求获取数据源信息
        res = await client.get("/datasources")
        # 将响应内容转换为 Result 对象
        result: Result = res.json()
        # 如果请求成功
        if result["success"]:
            # 返回数据源模型列表
            return [DatasourceModel(**datasource) for datasource in result["data"]]
        else:
            # 抛出客户端异常，包含错误状态码和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 抛出客户端异常，包含错误信息
        raise ClientException(f"Failed to list datasource: {e}")
```