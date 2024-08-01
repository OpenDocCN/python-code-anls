# `.\DB-GPT-src\dbgpt\serve\prompt\api\endpoints.py`

```py
from functools import cache  # 导入 functools 模块中的 cache 装饰器
from typing import List, Optional  # 导入 typing 模块中的 List 和 Optional 类型

from fastapi import (  # 导入 fastapi 模块中的 APIRouter、Depends、HTTPException、Query、Request
    APIRouter, Depends, HTTPException, Query, Request
)
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer  # 导入 fastapi.security.http 模块中的 HTTPAuthorizationCredentials、HTTPBearer 类

from dbgpt.component import SystemApp  # 从 dbgpt.component 模块导入 SystemApp 类
from dbgpt.serve.core import Result  # 从 dbgpt.serve.core 模块导入 Result 类
from dbgpt.util import PaginationResult  # 从 dbgpt.util 模块导入 PaginationResult 类

from ..config import (  # 从父级目录中导入 config 模块中的 APP_NAME、SERVE_APP_NAME、SERVE_SERVICE_COMPONENT_NAME、ServeConfig 类
    APP_NAME, SERVE_APP_NAME, SERVE_SERVICE_COMPONENT_NAME, ServeConfig
)
from ..service.service import Service  # 从父级目录中导入 service 模块中的 Service 类
from .schemas import ServeRequest, ServerResponse  # 从当前目录中的 schemas 模块导入 ServeRequest、ServerResponse 类

router = APIRouter()  # 创建一个 APIRouter 实例，用于定义路由和请求处理函数

# Add your API endpoints here

global_system_app: Optional[SystemApp] = None  # 定义一个全局变量 global_system_app，可能为 None


def get_service() -> Service:
    """Get the service instance"""
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)
    # 获取全局系统应用对象中 SERVE_SERVICE_COMPONENT_NAME 组件的 Service 实例


get_bearer_token = HTTPBearer(auto_error=False)  # 创建一个 HTTPBearer 实例，用于处理 HTTP Bearer Token


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
    # 解析逗号分隔的 API 密钥字符串为 API 密钥列表，并去除每个密钥的空格


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    request: Request = None,
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

    """
    if request.url.path.startswith(f"/api/v1"):
        return None
    # 如果请求路径以 "/api/v1" 开头，直接返回 None 表示允许访问

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
    # 如果服务配置中设置了 API 密钥，检查授权凭据是否在 API 密钥列表中，如果不在则抛出 HTTPException 异常


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}
    # 健康检查端点，返回状态为 "ok" 的字典


@router.get("/test_auth", dependencies=[Depends(check_api_key)])
async def test_auth():
    """Test auth endpoint"""
    return {"status": "ok"}
    # 测试授权端点，返回状态为 "ok" 的字典


# TODO: Compatible with old API, will be modified in the future
@router.post(
    "/add", response_model=Result[ServerResponse], dependencies=[Depends(check_api_key)]
)
async def create(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    # 添加数据端点，使用 ServeRequest 作为请求模型，返回 Result[ServerResponse] 类型的响应模型
    """创建一个新的 Prompt 实体

    Args:
        request (ServeRequest): 请求对象，包含创建 Prompt 所需的信息
        service (Service): 服务对象，用于执行创建 Prompt 的操作
    Returns:
        ServerResponse: 返回一个包含创建结果的响应对象
    """
    # 调用服务对象的 create 方法来创建 Prompt，并将结果封装成成功的响应对象返回
    return Result.succ(service.create(request))
@router.post(
    "/update",
    response_model=Result[ServerResponse],
    dependencies=[Depends(check_api_key)],
)
async def update(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """Update a Prompt entity

    Args:
        request (ServeRequest): The request object containing data to update
        service (Service): The service dependency providing business logic
    Returns:
        Result[ServerResponse]: Result object indicating success or failure of update operation
    """
    return Result.succ(service.update(request))



@router.post(
    "/delete", response_model=Result[None], dependencies=[Depends(check_api_key)]
)
async def delete(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[None]:
    """Delete a Prompt entity

    Args:
        request (ServeRequest): The request object containing data to delete
        service (Service): The service dependency providing business logic
    Returns:
        Result[None]: Result object indicating success or failure of delete operation
    """
    return Result.succ(service.delete(request))



@router.post(
    "/list",
    response_model=Result[List[ServerResponse]],
    dependencies=[Depends(check_api_key)],
)
async def query(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[List[ServerResponse]]:
    """Query Prompt entities

    Args:
        request (ServeRequest): The request object containing query parameters
        service (Service): The service dependency providing business logic
    Returns:
        Result[List[ServerResponse]]: Result object containing a list of entities as ServerResponse
    """
    return Result.succ(service.get_list(request))



@router.post(
    "/query_page",
    response_model=Result[PaginationResult[ServerResponse]],
    dependencies=[Depends(check_api_key)],
)
async def query_page(
    request: ServeRequest,
    page: Optional[int] = Query(default=1, description="current page"),
    page_size: Optional[int] = Query(default=20, description="page size"),
    service: Service = Depends(get_service),
) -> Result[PaginationResult[ServerResponse]]:
    """Query Prompt entities with pagination

    Args:
        request (ServeRequest): The request object containing query parameters
        page (int): The page number to fetch
        page_size (int): The number of entities per page
        service (Service): The service dependency providing business logic
    Returns:
        Result[PaginationResult[ServerResponse]]: Result object containing paginated entities
    """
    return Result.succ(service.get_list_by_page(request, page, page_size))



def init_endpoints(system_app: SystemApp) -> None:
    """Initialize the endpoints

    Args:
        system_app (SystemApp): The application instance to register endpoints with
    """
    global global_system_app
    system_app.register(Service)
    global_system_app = system_app
```