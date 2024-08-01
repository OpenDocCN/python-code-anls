# `.\DB-GPT-src\dbgpt\serve\datasource\api\endpoints.py`

```py
from functools import cache
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from dbgpt.component import SystemApp
from dbgpt.serve.core import Result
from dbgpt.serve.datasource.api.schemas import (
    DatasourceServeRequest,
    DatasourceServeResponse,
)
from dbgpt.serve.datasource.config import SERVE_SERVICE_COMPONENT_NAME
from dbgpt.serve.datasource.service.service import Service
from dbgpt.util import PaginationResult

router = APIRouter()

# 全局变量，用于存储系统应用的实例，初始值为 None
global_system_app: Optional[SystemApp] = None


def get_service() -> Service:
    """获取服务实例"""
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)


get_bearer_token = HTTPBearer(auto_error=False)


@cache
def _parse_api_keys(api_keys: str) -> List[str]:
    """解析字符串形式的 API 密钥为列表

    Args:
        api_keys (str): 字符串形式的 API 密钥

    Returns:
        List[str]: API 密钥列表
    """
    if not api_keys:
        return []
    return [key.strip() for key in api_keys.split(",")]


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    service: Service = Depends(get_service),
) -> Optional[str]:
    """检查 API 密钥

    如果未设置 API 密钥，则允许所有请求。

    您可以在请求头中传递令牌，如下所示：

    .. code-block:: python

        import requests

        client_api_key = "your_api_key"
        headers = {"Authorization": "Bearer " + client_api_key}
        res = requests.get("http://test/hello", headers=headers)
        assert res.status_code == 200

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
        # 如果未设置 api_keys，则允许所有请求
        return None


@router.get("/health", dependencies=[Depends(check_api_key)])
async def health():
    """健康检查端点"""
    return {"status": "ok"}


@router.get("/test_auth", dependencies=[Depends(check_api_key)])
async def test_auth():
    """测试认证端点"""
    return {"status": "ok"}


@router.post("/datasources", dependencies=[Depends(check_api_key)])
async def create(
    request: DatasourceServeRequest, service: Service = Depends(get_service)
) -> Result:
    """创建新的数据源实体

    Args:
        request (DatasourceServeRequest): 请求对象
        service (Service): 服务对象
    """
    # 返回从服务中创建请求后的成功结果
    return Result.succ(service.create(request))
@router.put("/datasources", dependencies=[Depends(check_api_key)])
async def update(
    request: DatasourceServeRequest, service: Service = Depends(get_service)
) -> Result:
    """Update a Space entity

    Args:
        request (DatasourceServeRequest): The request object containing data to update
        service (Service): The service dependency used to perform the update operation
    Returns:
        Result: Success or failure result of the update operation
    """
    # 调用服务的更新方法，并返回结果
    return Result.succ(service.update(request))


@router.delete(
    "/datasources/{datasource_id}",
    response_model=Result[None],
    dependencies=[Depends(check_api_key)],
)
async def delete(
    datasource_id: str, service: Service = Depends(get_service)
) -> Result[None]:
    """Delete a Space entity

    Args:
        datasource_id (str): The ID of the datasource entity to delete
        service (Service): The service dependency used to perform the delete operation
    Returns:
        Result[None]: Success or failure result of the delete operation
    """
    # 调用服务的删除方法，并返回结果
    return Result.succ(service.delete(datasource_id))


@router.get(
    "/datasources/{datasource_id}",
    dependencies=[Depends(check_api_key)],
    response_model=Result[List],
)
async def query(
    datasource_id: str, service: Service = Depends(get_service)
) -> Result[List[DatasourceServeResponse]]:
    """Query Space entities by datasource ID

    Args:
        datasource_id (str): The ID of the datasource entity to query
        service (Service): The service dependency used to perform the query operation
    Returns:
        Result[List[DatasourceServeResponse]]: Success or failure result containing a list of datasource responses
    """
    # 调用服务的获取方法，并返回结果
    return Result.succ(service.get(datasource_id))


@router.get(
    "/datasources",
    dependencies=[Depends(check_api_key)],
    response_model=Result[PaginationResult[DatasourceServeResponse]],
)
async def query_page(
    page: int = Query(default=1, description="current page"),
    page_size: int = Query(default=20, description="page size"),
    service: Service = Depends(get_service),
) -> Result[PaginationResult[DatasourceServeResponse]]:
    """Query Space entities with pagination

    Args:
        page (int): The page number to retrieve
        page_size (int): The number of items per page
        service (Service): The service dependency used to perform the pagination query
    Returns:
        Result[PaginationResult[DatasourceServeResponse]]: Success or failure result containing paginated datasource responses
    """
    # 调用服务的列表方法，并返回结果
    return Result.succ(service.list())


def init_endpoints(system_app: SystemApp) -> None:
    """Initialize the endpoints

    Args:
        system_app (SystemApp): The application instance to register the service with
    """
    # 注册服务到系统应用中
    global global_system_app
    system_app.register(Service)
    global_system_app = system_app
```