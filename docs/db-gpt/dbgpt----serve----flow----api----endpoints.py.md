# `.\DB-GPT-src\dbgpt\serve\flow\api\endpoints.py`

```py
from functools import cache
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

from dbgpt.component import SystemApp
from dbgpt.core.awel.flow import ResourceMetadata, ViewMetadata
from dbgpt.serve.core import Result
from dbgpt.util import PaginationResult

from ..config import APP_NAME, SERVE_SERVICE_COMPONENT_NAME, ServeConfig
from ..service.service import Service
from .schemas import ServeRequest, ServerResponse

# 创建一个名为 router 的 APIRouter 实例
router = APIRouter()

# 全局变量，用于存储 SystemApp 的实例，可选类型为 SystemApp 或 None
global_system_app: Optional[SystemApp] = None


def get_service() -> Service:
    """获取 Service 的实例"""
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)


# 定义一个获取 bearer token 的依赖项，使用 HTTPBearer 进行验证，不强制错误返回
get_bearer_token = HTTPBearer(auto_error=False)


@cache
def _parse_api_keys(api_keys: str) -> List[str]:
    """解析字符串形式的 API key 到列表

    Args:
        api_keys (str): 字符串形式的 API key

    Returns:
        List[str]: API key 列表
    """
    if not api_keys:
        return []
    return [key.strip() for key in api_keys.split(",")]


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    request: Request = None,
    service: Service = Depends(get_service),
) -> Optional[str]:
    """检查 API key 是否有效

    如果 API key 未设置，允许所有请求。

    你可以通过请求头传递 token，如下所示：

    .. code-block:: python

        import requests

        client_api_key = "your_api_key"
        headers = {"Authorization": "Bearer " + client_api_key}
        res = requests.get("http://test/hello", headers=headers)
        assert res.status_code == 200

    """
    if request.url.path.startswith(f"/api/v1"):
        return None

    # 检查服务配置中是否存在 API keys，并解析成列表形式
    if service.config.api_keys:
        api_keys = _parse_api_keys(service.config.api_keys)
        # 如果没有传入认证信息或传入的 token 不在允许的 API keys 中，则抛出 HTTPException
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
        # 如果 API keys 没有设置，则允许所有请求
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
    "/flows", response_model=Result[None], dependencies=[Depends(check_api_key)]
)
async def create(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """创建新的 Flow 实体

    Args:
        request (ServeRequest): 请求对象，包含创建 Flow 实体所需的数据
        service (Service): Service 的实例

    Returns:
        Result[ServerResponse]: 包含操作结果的 Result 对象，包括创建后的服务器响应
    """
    Args:
        request (ServeRequest): 表示传入的请求对象，类型为 ServeRequest
        service (Service): 表示传入的服务对象，类型为 Service
    Returns:
        ServerResponse: 返回一个服务器响应对象，类型为 ServerResponse
    """
    # 调用 service 对象的 create_and_save_dag 方法来创建并保存 DAG，然后封装在 Result.succ 中返回
    return Result.succ(service.create_and_save_dag(request))
@router.put(
    "/flows/{uid}",
    response_model=Result[ServerResponse],
    dependencies=[Depends(check_api_key)],
)
async def update(
    uid: str, request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """Update a Flow entity

    Args:
        uid (str): The uid identifying the flow entity to update
        request (ServeRequest): The request containing updated data
        service (Service): The service instance used for updating
    Returns:
        ServerResponse: The response containing the result of the update operation
    """
    return Result.succ(service.update_flow(request))


@router.delete("/flows/{uid}")
async def delete(
    uid: str, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """Delete a Flow entity

    Args:
        uid (str): The uid identifying the flow entity to delete
        service (Service): The service instance used for deletion
    Returns:
        Result[None]: The response indicating success or failure of deletion
    """
    inst = service.delete(uid)
    return Result.succ(inst)


@router.get("/flows/{uid}")
async def get_flows(
    uid: str, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """Get a Flow entity by uid

    Args:
        uid (str): The uid identifying the flow entity to retrieve
        service (Service): The service instance used for retrieval

    Returns:
        Result[ServerResponse]: The response containing the retrieved flow entity
    """
    flow = service.get({"uid": uid})
    if not flow:
        raise HTTPException(status_code=404, detail=f"Flow {uid} not found")
    return Result.succ(flow)


@router.get(
    "/flows",
    response_model=Result[PaginationResult[ServerResponse]],
    dependencies=[Depends(check_api_key)],
)
async def query_page(
    user_name: Optional[str] = Query(default=None, description="user name"),
    sys_code: Optional[str] = Query(default=None, description="system code"),
    page: int = Query(default=1, description="current page"),
    page_size: int = Query(default=20, description="page size"),
    name: Optional[str] = Query(default=None, description="flow name"),
    uid: Optional[str] = Query(default=None, description="flow uid"),
    service: Service = Depends(get_service),
) -> Result[PaginationResult[ServerResponse]]:
    """Query Flow entities by various parameters

    Args:
        user_name (Optional[str]): Filter by username
        sys_code (Optional[str]): Filter by system code
        page (int): The page number for pagination
        page_size (int): The number of items per page
        name (Optional[str]): Filter by flow name
        uid (Optional[str]): Filter by flow uid
        service (Service): The service instance used for querying
    Returns:
        ServerResponse: The response containing the paginated list of flow entities
    """
    return Result.succ(
        service.get_list_by_page(
            {"user_name": user_name, "sys_code": sys_code, "name": name, "uid": uid},
            page,
            page_size,
        )
    )


@router.get("/nodes", dependencies=[Depends(check_api_key)])
async def get_nodes() -> Result[List[Union[ViewMetadata, ResourceMetadata]]]:
    """Get the list of operator or resource nodes available in the system

    Returns:
        Result[List[Union[ViewMetadata, ResourceMetadata]]]: The list of nodes
    """
    from dbgpt.core.awel.flow.base import _OPERATOR_REGISTRY
    # 返回一个成功的结果，结果的数据是 _OPERATOR_REGISTRY 的 metadata_list() 方法返回的内容
    return Result.succ(_OPERATOR_REGISTRY.metadata_list())
# 初始化端点函数，将指定的系统应用程序注册到全局系统应用程序变量中
def init_endpoints(system_app: SystemApp) -> None:
    """Initialize the endpoints"""
    # 声明全局变量 global_system_app，用于存储系统应用程序实例
    global global_system_app
    # 调用系统应用程序的 register 方法，注册一个服务（Service）到系统应用程序中
    system_app.register(Service)
    # 将传入的系统应用程序实例赋值给全局系统应用程序变量
    global_system_app = system_app
```