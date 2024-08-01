# `.\DB-GPT-src\dbgpt\serve\rag\api\endpoints.py`

```py
from functools import cache
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from dbgpt.component import SystemApp
from dbgpt.serve.core import Result
from dbgpt.serve.rag.api.schemas import (
    DocumentServeRequest,
    DocumentServeResponse,
    KnowledgeSyncRequest,
    SpaceServeRequest,
    SpaceServeResponse,
)
from dbgpt.serve.rag.config import SERVE_SERVICE_COMPONENT_NAME
from dbgpt.serve.rag.service.service import Service
from dbgpt.util import PaginationResult

# Define a new APIRouter instance for defining endpoints
router = APIRouter()

# Global variable to hold the SystemApp instance, which is optional
global_system_app: Optional[SystemApp] = None


def get_service() -> Service:
    """Get the service instance
    
    Returns:
        Service: The instance of the Service class
    """
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)


# Function to parse API keys from a string to a list
@cache
def _parse_api_keys(api_keys: str) -> List[str]:
    """Parse the string api keys to a list

    Args:
        api_keys (str): The string api keys

    Returns:
        List[str]: The list of api keys
    """
    if not api_keys:
        return []
    return [key.strip() for key in api_keys.split(",")]


# Dependency function to check the validity of the API key
async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    service: Service = Depends(get_service),
) -> Optional[str]:
    """Check the api key

    If the api key is not set, allow all.

    Your can pass the token in you request header like this:

    .. code-block:: python

        import requests

        client_api_key = "your_api_key"
        headers = {"Authorization": "Bearer " + client_api_key}
        res = requests.get("http://test/hello", headers=headers)
        assert res.status_code == 200

    Args:
        auth (Optional[HTTPAuthorizationCredentials], optional): Authorization credentials. Defaults to Depends(get_bearer_token).
        service (Service, optional): Instance of the Service class. Defaults to Depends(get_service).

    Returns:
        Optional[str]: The API key token if valid, otherwise None
    """
    if service.config.api_keys:
        api_keys = _parse_api_keys(service.config.api_keys)
        if auth is None or (token := auth.credentials) not in api_keys:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:
        # api_keys not set; allow all
        return None


# Endpoint to perform a health check
@router.get("/health", dependencies=[Depends(check_api_key)])
async def health():
    """Health check endpoint
    
    Returns:
        dict: A dictionary indicating the status of the service
    """
    return {"status": "ok"}


# Endpoint to test authentication
@router.get("/test_auth", dependencies=[Depends(check_api_key)])
async def test_auth():
    """Test auth endpoint
    
    Returns:
        dict: A dictionary indicating the authentication status
    """
    return {"status": "ok"}


# Endpoint to create a new Space entity
@router.post("/spaces", dependencies=[Depends(check_api_key)])
async def create(
    request: SpaceServeRequest, service: Service = Depends(get_service)
) -> Result:
    """Create a new Space entity
    
    Args:
        request (SpaceServeRequest): Request body containing data for creating a new Space entity
        service (Service, optional): Instance of the Service class. Defaults to Depends(get_service).

    Returns:
        Result: Result object indicating the outcome of the creation operation
    """
    # Implementation of the create endpoint will follow in the next part of the code block
    Args:
        request (SpaceServeRequest): 表示传入的请求对象，类型为 SpaceServeRequest
        service (Service): 表示传入的服务对象，类型为 Service
    Returns:
        ServerResponse: 表示返回的服务器响应对象，类型为 ServerResponse
    """
    # 调用 service 对象的 create_space 方法，以 request 作为参数，创建空间并返回成功的结果对象
    return Result.succ(service.create_space(request))
@router.put("/spaces", dependencies=[Depends(check_api_key)])
async def update(
    request: SpaceServeRequest, service: Service = Depends(get_service)
) -> Result:
    """Update a Space entity

    Args:
        request (SpaceServeRequest): The request object containing details to update
        service (Service): The service dependency used to perform the update
    Returns:
        Result: A Result object indicating success or failure of the update operation
    """
    return Result.succ(service.update_space(request))


@router.delete(
    "/spaces/{space_id}",
    response_model=Result[None],
    dependencies=[Depends(check_api_key)],
)
async def delete(
    space_id: str, service: Service = Depends(get_service)
) -> Result[None]:
    """Delete a Space entity by its ID

    Args:
        space_id (str): The unique identifier of the Space entity to delete
        service (Service): The service dependency used to perform the delete operation
    Returns:
        Result[None]: A Result object indicating success or failure of the delete operation
    """
    return Result.succ(service.delete(space_id))


@router.get(
    "/spaces/{space_id}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[List],
)
async def query(
    space_id: str, service: Service = Depends(get_service)
) -> Result[List[SpaceServeResponse]]:
    """Retrieve a Space entity by its ID

    Args:
        space_id (str): The unique identifier of the Space entity to retrieve
        service (Service): The service dependency used to perform the retrieval operation
    Returns:
        Result[List[SpaceServeResponse]]: A Result object containing a list of SpaceServeResponse objects
    """
    request = {"id": space_id}
    return Result.succ(service.get(request))


@router.get(
    "/spaces",
    dependencies=[Depends(check_api_key)],
    response_model=Result[PaginationResult[SpaceServeResponse]],
)
async def query_page(
    page: int = Query(default=1, description="current page"),
    page_size: int = Query(default=20, description="page size"),
    service: Service = Depends(get_service),
) -> Result[PaginationResult[SpaceServeResponse]]:
    """Query Space entities with pagination

    Args:
        page (int): The page number to retrieve
        page_size (int): The number of items per page
        service (Service): The service dependency used to perform the query operation
    Returns:
        Result[PaginationResult[SpaceServeResponse]]: A Result object containing a paginated list of SpaceServeResponse objects
    """
    return Result.succ(service.get_list_by_page({}, page, page_size))


@router.post("/documents", dependencies=[Depends(check_api_key)])
async def create_document(
    doc_name: str = Form(...),
    doc_type: str = Form(...),
    space_id: str = Form(...),
    content: Optional[str] = Form(None),
    doc_file: Optional[UploadFile] = File(None),
    service: Service = Depends(get_service),
) -> Result:
    """Create a new Document entity

    Args:
        doc_name (str): The name of the document
        doc_type (str): The type of the document
        space_id (str): The ID of the space where the document belongs
        content (Optional[str]): The content of the document (optional)
        doc_file (Optional[UploadFile]): The file of the document (optional)
        service (Service): The service dependency used to perform the create operation
    Returns:
        Result: A Result object indicating success or failure of the create operation
    """
    request = DocumentServeRequest(
        doc_name=doc_name,
        doc_type=doc_type,
        content=content,
        doc_file=doc_file,
        space_id=space_id,
    )
    return Result.succ(await service.create_document(request))


@router.get(
    "/documents/{document_id}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[List],
)
async def query_document(
    document_id: str, service: Service = Depends(get_service)
) -> Result[List]:
    """Query a Document entity by its ID

    Args:
        document_id (str): The unique identifier of the Document entity to retrieve
        service (Service): The service dependency used to perform the retrieval operation
    Returns:
        Result[List]: A Result object containing a list of Document entities matching the ID
    """
    return Result.succ(service.get_document(document_id))
# 定义一个异步函数，用于查询空间实体的详细信息
async def query_page(
    page: int = Query(default=1, description="current page"),  # 定义名为page的参数，默认值为1，描述为当前页码
    page_size: int = Query(default=20, description="page size"),  # 定义名为page_size的参数，默认值为20，描述为每页的条目数
    service: Service = Depends(get_service),  # 依赖注入，获取服务对象Service
) -> Result[PaginationResult[DocumentServeResponse]]:
    """Query Space entities

    Args:
        page (int): The page number
        page_size (int): The page size
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    # 调用服务对象的get_document_list方法，查询空间实体列表，并返回成功的结果对象
    return Result.succ(service.get_document_list({}, page, page_size))


@router.post("/documents/sync", dependencies=[Depends(check_api_key)])
async def sync_documents(
    requests: List[KnowledgeSyncRequest],  # 请求参数，包含KnowledgeSyncRequest类型的列表
    service: Service = Depends(get_service)  # 依赖注入，获取服务对象Service
) -> Result:
    """Create a new Document entity

    Args:
        request (SpaceServeRequest): The request
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    # 调用服务对象的sync_document方法，同步创建文档实体，并返回成功的结果对象
    return Result.succ(service.sync_document(requests))


@router.delete(
    "/documents/{document_id}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[None],
)
async def delete_document(
    document_id: str,  # 请求参数，文档ID，字符串类型
    service: Service = Depends(get_service)  # 依赖注入，获取服务对象Service
) -> Result[None]:
    """Delete a Space entity

    Args:
        request (SpaceServeRequest): The request
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    # 调用服务对象的delete_document方法，删除指定ID的文档实体，并返回成功的结果对象
    return Result.succ(service.delete_document(document_id))


def init_endpoints(system_app: SystemApp) -> None:
    """Initialize the endpoints"""
    global global_system_app  # 声明全局变量global_system_app，用于存储系统应用对象
    system_app.register(Service)  # 注册服务对象到系统应用
    global_system_app = system_app  # 将传入的系统应用对象赋值给全局变量global_system_app
```