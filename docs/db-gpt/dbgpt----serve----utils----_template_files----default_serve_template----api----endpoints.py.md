# `.\DB-GPT-src\dbgpt\serve\utils\_template_files\default_serve_template\api\endpoints.py`

```py
# 从 functools 模块导入 cache 装饰器，用于函数结果的缓存
from functools import cache
# 从 typing 模块导入 List（列表）和 Optional（可选类型）
from typing import List, Optional

# 从 fastapi 框架导入 APIRouter（API 路由器）、Depends（依赖注入）、HTTPException（HTTP 异常）、Query（查询参数）
from fastapi import APIRouter, Depends, HTTPException, Query
# 从 fastapi.security.http 模块导入 HTTPAuthorizationCredentials（HTTP 授权凭证）、HTTPBearer（HTTP Bearer Token 验证）
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

# 从 dbgpt.component 模块导入 SystemApp（系统应用）
from dbgpt.component import SystemApp
# 从 dbgpt.serve.core 模块导入 Result（结果类型）
from dbgpt.serve.core import Result
# 从 dbgpt.util 模块导入 PaginationResult（分页结果类型）

# 从当前包的 config 模块导入 APP_NAME（应用名称）、SERVE_APP_NAME（服务应用名称）、SERVE_SERVICE_COMPONENT_NAME（服务组件名称）、ServeConfig（服务配置）
from ..config import APP_NAME, SERVE_APP_NAME, SERVE_SERVICE_COMPONENT_NAME, ServeConfig
# 从当前包的 service 模块导入 Service（服务类）
from ..service.service import Service
# 从当前包的 schemas 模块导入 ServeRequest（服务请求模型）、ServerResponse（服务器响应模型）

# 创建一个 APIRouter 实例
router = APIRouter()

# 全局变量，用于存储 SystemApp 实例或者为空
global_system_app: Optional[SystemApp] = None


def get_service() -> Service:
    """获取服务实例"""
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)


# 获取 Bearer Token 的依赖注入函数，自动处理错误
get_bearer_token = HTTPBearer(auto_error=False)


@cache
def _parse_api_keys(api_keys: str) -> List[str]:
    """解析字符串类型的 API 密钥为列表

    Args:
        api_keys (str): 字符串类型的 API 密钥

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

    如果 API 密钥未设置，则允许所有请求。

    可以通过请求头部传递 Token，示例：

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
            # 如果未提供正确的 API 密钥，则返回 HTTP 401 错误
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
        # 如果未设置 API 密钥，则允许所有请求
        return None


@router.get("/health")
async def health():
    """健康检查端点"""
    return {"status": "ok"}


@router.get("/test_auth", dependencies=[Depends(check_api_key)])
async def test_auth():
    """测试认证端点"""
    return {"status": "ok"}


@router.post(
    "/", response_model=Result[ServerResponse], dependencies=[Depends(check_api_key)]
)
async def create(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """创建一个新的 {__template_app_name__hump__} 实体

    Args:
        request (ServeRequest): 请求对象
        service (Service): 服务对象
    Returns:
        ServerResponse: 响应对象
    """
    # 调用 service 的 create 方法，并将 request 作为参数传递，然后将返回的结果封装成一个成功的 Result 对象并返回
    return Result.succ(service.create(request))
@router.put(
    "/", response_model=Result[ServerResponse], dependencies=[Depends(check_api_key)]
)
async def update(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """Update a {__template_app_name__hump__} entity

    Args:
        request (ServeRequest): The request object containing update details
        service (Service): The service instance providing update functionality
    Returns:
        ServerResponse: The response containing the result of the update operation
    """
    return Result.succ(service.update(request))


@router.post(
    "/query",
    response_model=Result[ServerResponse],
    dependencies=[Depends(check_api_key)],
)
async def query(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """Query {__template_app_name__hump__} entities

    Args:
        request (ServeRequest): The request object containing query parameters
        service (Service): The service instance providing query functionality
    Returns:
        ServerResponse: The response containing the result of the query operation
    """
    return Result.succ(service.get(request))


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
    """Query {__template_app_name__hump__} entities with pagination

    Args:
        request (ServeRequest): The request object containing pagination parameters
        page (int): The page number to retrieve
        page_size (int): The number of items per page
        service (Service): The service instance providing pagination query functionality
    Returns:
        ServerResponse: The response containing the paginated result of the query
    """
    return Result.succ(service.get_list_by_page(request, page, page_size))


def init_endpoints(system_app: SystemApp) -> None:
    """Initialize the endpoints for the system application

    Args:
        system_app (SystemApp): The system application instance
    """
    global global_system_app
    # Register the Service class with the system application
    system_app.register(Service)
    # Set the global system application variable
    global_system_app = system_app
```