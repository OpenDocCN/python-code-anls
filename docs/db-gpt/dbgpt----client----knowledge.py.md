# `.\DB-GPT-src\dbgpt\client\knowledge.py`

```py
"""Knowledge API client."""
# 引入必要的模块和类型提示
import json
from typing import List

# 引入内部模块和类
from dbgpt._private.pydantic import model_to_dict, model_to_json
from dbgpt.core.schema.api import Result

# 引入自定义的客户端和异常类
from .client import Client, ClientException
from .schema import DocumentModel, SpaceModel, SyncModel


async def create_space(client: Client, space_model: SpaceModel) -> SpaceModel:
    """Create a new space.

    Args:
        client (Client): The dbgpt client.
        space_model (SpaceModel): The space model.
    Returns:
        SpaceModel: The space model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发送 POST 请求以创建新的空间，将 space_model 转换为字典格式
        res = await client.post("/knowledge/spaces", model_to_dict(space_model))
        # 解析响应并转换为 Result 对象
        result: Result = res.json()
        # 如果请求成功，则返回包含数据的 SpaceModel 对象；否则抛出异常
        if result["success"]:
            return SpaceModel(**result["data"])
        else:
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，并将其包装为 ClientException 异常抛出
        raise ClientException(f"Failed to create space: {e}")


async def update_space(client: Client, space_model: SpaceModel) -> SpaceModel:
    """Update a document.

    Args:
        client (Client): The dbgpt client.
        space_model (SpaceModel): The space model.
    Returns:
        SpaceModel: The space model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发送 PUT 请求以更新空间信息，将 space_model 转换为字典格式
        res = await client.put("/knowledge/spaces", model_to_dict(space_model))
        # 解析响应并转换为 Result 对象
        result: Result = res.json()
        # 如果请求成功，则返回更新后的 SpaceModel 对象；否则抛出异常
        if result["success"]:
            return SpaceModel(**result["data"])
        else:
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，并将其包装为 ClientException 异常抛出
        raise ClientException(f"Failed to update space: {e}")


async def delete_space(client: Client, space_id: str) -> SpaceModel:
    """Delete a space.

    Args:
        client (Client): The dbgpt client.
        space_id (str): The space id.
    Returns:
        SpaceModel: The space model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发送 DELETE 请求以删除指定空间，拼接 space_id 到 URL 中
        res = await client.delete("/knowledge/spaces/" + space_id)
        # 解析响应并转换为 Result 对象
        result: Result = res.json()
        # 如果请求成功，则返回被删除的 SpaceModel 对象；否则抛出异常
        if result["success"]:
            return SpaceModel(**result["data"])
        else:
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，并将其包装为 ClientException 异常抛出
        raise ClientException(f"Failed to delete space: {e}")


async def get_space(client: Client, space_id: str) -> SpaceModel:
    """Get a document.

    Args:
        client (Client): The dbgpt client.
        space_id (str): The space id.
    Returns:
        SpaceModel: The space model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发送 GET 请求以获取指定空间的信息，拼接 space_id 到 URL 中
        res = await client.get("/knowledge/spaces/" + space_id)
        # 解析响应并转换为 Result 对象
        result: Result = res.json()
        # 如果请求成功，则返回获取的 SpaceModel 对象；否则抛出异常
        if result["success"]:
            return SpaceModel(**result["data"])
        else:
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，并将其包装为 ClientException 异常抛出
        raise ClientException(f"Failed to get space: {e}")
    # 捕获任何异常并将其作为客户端异常抛出，包含失败原因
    except Exception as e:
        raise ClientException(f"Failed to get space: {e}")
async def list_space(client: Client) -> List[SpaceModel]:
    """List spaces.

    Args:
        client (Client): The dbgpt client.
    Returns:
        List[SpaceModel]: The list of space models.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起异步请求，获取空间列表
        res = await client.get("/knowledge/spaces")
        # 将响应内容解析为 Result 对象
        result: Result = res.json()
        # 检查请求是否成功
        if result["success"]:
            # 如果成功，返回由 SpaceModel 对象组成的列表
            return [SpaceModel(**space) for space in result["data"]["items"]]
        else:
            # 如果请求失败，抛出 ClientException 异常，包含错误状态和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，抛出 ClientException 异常，包含错误信息
        raise ClientException(f"Failed to list spaces: {e}")


async def create_document(client: Client, doc_model: DocumentModel) -> DocumentModel:
    """Create a new document.

    Args:
        client (Client): The dbgpt client.
        doc_model (SpaceModel): The document model.

    """
    try:
        # 发起异步 POST 请求，创建新文档，使用 model_to_dict 将 DocumentModel 转为字典形式
        res = await client.post_param("/knowledge/documents", model_to_dict(doc_model))
        # 将响应内容解析为 Result 对象
        result: Result = res.json()
        # 检查请求是否成功
        if result["success"]:
            # 如果成功，返回新创建的 DocumentModel 对象
            return DocumentModel(**result["data"])
        else:
            # 如果请求失败，抛出 ClientException 异常，包含错误状态和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，抛出 ClientException 异常，包含错误信息
        raise ClientException(f"Failed to create document: {e}")


async def delete_document(client: Client, document_id: str) -> DocumentModel:
    """Delete a document.

    Args:
        client (Client): The dbgpt client.
        document_id (str): The document id.
    Returns:
        DocumentModel: The document model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起异步 DELETE 请求，删除指定 ID 的文档
        res = await client.delete("/knowledge/documents/" + document_id)
        # 将响应内容解析为 Result 对象
        result: Result = res.json()
        # 检查请求是否成功
        if result["success"]:
            # 如果成功，返回被删除的 DocumentModel 对象
            return DocumentModel(**result["data"])
        else:
            # 如果请求失败，抛出 ClientException 异常，包含错误状态和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，抛出 ClientException 异常，包含错误信息
        raise ClientException(f"Failed to delete document: {e}")


async def get_document(client: Client, document_id: str) -> DocumentModel:
    """Get a document.

    Args:
        client (Client): The dbgpt client.
        document_id (str): The document id.
    Returns:
        DocumentModel: The document model.
    Raises:
        ClientException: If the request failed.
    """
    try:
        # 发起异步 GET 请求，获取指定 ID 的文档
        res = await client.get("/knowledge/documents/" + document_id)
        # 将响应内容解析为 Result 对象
        result: Result = res.json()
        # 检查请求是否成功
        if result["success"]:
            # 如果成功，返回获取的 DocumentModel 对象
            return DocumentModel(**result["data"])
        else:
            # 如果请求失败，抛出 ClientException 异常，包含错误状态和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常，抛出 ClientException 异常，包含错误信息
        raise ClientException(f"Failed to get document: {e}")


async def list_document(client: Client) -> List[DocumentModel]:
    """List documents.

    Args:
        client (Client): The dbgpt client.
    """
    # 尝试从客户端获取"/knowledge/documents"的内容
    try:
        # 发起异步请求获取数据
        res = await client.get("/knowledge/documents")
        # 将响应内容解析为Result对象
        result: Result = res.json()
        # 检查返回结果中的success字段
        if result["success"]:
            # 如果成功，将每个文档条目转换为DocumentModel对象并返回列表
            return [DocumentModel(**document) for document in result["data"]["items"]]
        else:
            # 如果不成功，抛出客户端异常，传递错误码和原因
            raise ClientException(status=result["err_code"], reason=result)
    except Exception as e:
        # 捕获所有异常情况，抛出客户端异常，提示失败原因
        raise ClientException(f"Failed to list documents: {e}")
async def sync_document(client: Client, sync_model: SyncModel) -> List:
    """Sync document.

    Args:
        client (Client): The dbgpt client.  # 接受一个 dbgpt 客户端对象作为参数
        sync_model (SyncModel): The sync model.  # 接受一个 SyncModel 对象作为参数，用于同步文档
    Returns:
        List: The list of document ids.  # 返回一个文档 id 的列表
    Raises:
        ClientException: If the request failed.  # 如果请求失败，则抛出 ClientException 异常
    """
    try:
        res = await client.post(
            "/knowledge/documents/sync", [json.loads(model_to_json(sync_model))]
        )
        result: Result = res.json()  # 将响应解析为 Result 对象
        if result["success"]:  # 如果请求成功
            return result["data"]  # 返回响应数据中的文档 id 列表
        else:
            raise ClientException(status=result["err_code"], reason=result)  # 如果请求失败，则抛出 ClientException 异常
    except Exception as e:
        raise ClientException(f"Failed to sync documents: {e}")  # 捕获所有异常并抛出 ClientException 异常，指示同步文档失败
```