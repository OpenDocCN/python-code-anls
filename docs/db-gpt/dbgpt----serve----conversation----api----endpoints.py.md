# `.\DB-GPT-src\dbgpt\serve\conversation\api\endpoints.py`

```py
# 导入模块uuid，用于生成唯一标识符
import uuid
# 导入functools模块中的cache装饰器，用于缓存函数的返回值
from functools import cache
# 导入List和Optional类型，以及HTTPException、Query、Request等类和函数
from typing import List, Optional

# 导入FastAPI框架的APIRouter类、Depends函数、HTTPException异常、Query函数、Request类
from fastapi import APIRouter, Depends, HTTPException, Query, Request
# 导入HTTPAuthorizationCredentials和HTTPBearer类，用于处理HTTP Bearer Token认证
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer

# 导入dbgpt.component模块中的SystemApp类
from dbgpt.component import SystemApp
# 导入dbgpt.serve.core模块中的Result类
from dbgpt.serve.core import Result
# 导入dbgpt.util模块中的PaginationResult类
from dbgpt.util import PaginationResult

# 从当前包的config模块中导入APP_NAME、SERVE_APP_NAME、SERVE_SERVICE_COMPONENT_NAME、ServeConfig等变量和类
from ..config import APP_NAME, SERVE_APP_NAME, SERVE_SERVICE_COMPONENT_NAME, ServeConfig
# 从当前包的service模块中导入Service类
from ..service.service import Service
# 从当前包的schemas模块中导入MessageVo类、ServeRequest类、ServerResponse类
from .schemas import MessageVo, ServeRequest, ServerResponse

# 创建一个APIRouter实例
router = APIRouter()

# Add your API endpoints here

# 全局变量，用于存储SystemApp的实例，初始值为None
global_system_app: Optional[SystemApp] = None


def get_service() -> Service:
    """获取Service的实例"""
    return global_system_app.get_component(SERVE_SERVICE_COMPONENT_NAME, Service)


# 获取Bearer Token的依赖函数，使用HTTPBearer进行认证，auto_error参数为False表示认证失败时不抛出异常
get_bearer_token = HTTPBearer(auto_error=False)


@cache
def _parse_api_keys(api_keys: str) -> List[str]:
    """解析字符串类型的api keys为列表

    Args:
        api_keys (str): 字符串类型的api keys

    Returns:
        List[str]: api keys组成的列表
    """
    if not api_keys:
        return []
    return [key.strip() for key in api_keys.split(",")]


async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(get_bearer_token),
    request: Request = None,
    service: Service = Depends(get_service),
) -> Optional[str]:
    """检查api key

    如果未设置api key，允许所有请求。

    可以通过请求头中的Authorization字段传递token，示例如下：

    .. code-block:: python

        import requests

        client_api_key = "your_api_key"
        headers = {"Authorization": "Bearer " + client_api_key}
        res = requests.get("http://test/hello", headers=headers)
        assert res.status_code == 200

    """
    if request.url.path.startswith(f"/api/v1"):
        return None

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
        # 未设置api keys，允许所有请求
        return None


@router.get("/health")
async def health():
    """健康检查接口"""
    return {"status": "ok"}


@router.get("/test_auth", dependencies=[Depends(check_api_key)])
async def test_auth():
    """测试认证接口"""
    return {"status": "ok"}


@router.post(
    "/query",
    response_model=Result[ServerResponse],
    dependencies=[Depends(check_api_key)],
)
async def query(
    request: ServeRequest, service: Service = Depends(get_service)
) -> Result[ServerResponse]:
    """查询Conversation实体
    Args:
@router.post(
    "/new",
    response_model=Result[ServerResponse],
    dependencies=[Depends(check_api_key)],
)
async def dialogue_new(
    chat_mode: str = "chat_normal",
    user_name: str = None,
    # TODO remove user id
    user_id: str = None,
    sys_code: str = None,
):
    # 如果未提供用户昵称，则使用用户ID作为昵称
    user_name = user_name or user_id
    # 创建一个唯一的会话ID
    unique_id = uuid.uuid1()
    # 创建一个服务器响应对象
    res = ServerResponse(
        user_input="",
        conv_uid=str(unique_id),
        chat_mode=chat_mode,
        user_name=user_name,
        sys_code=sys_code,
    )
    # 返回成功的结果和响应对象
    return Result.succ(res)


@router.post(
    "/delete",
    dependencies=[Depends(check_api_key)],
)
async def delete(con_uid: str, service: Service = Depends(get_service)):
    """Delete a Conversation entity

    Args:
        con_uid (str): The conversation UID
        service (Service): The service
    """
    # 调用服务对象的删除方法，删除指定会话UID的会话实体
    service.delete(ServeRequest(conv_uid=con_uid))
    # 返回成功的结果
    return Result.succ(None)


@router.post(
    "/query_page",
    response_model=Result[PaginationResult[ServerResponse]],
    dependencies=[Depends(check_api_key)],
)
async def query_page(
    request: ServeRequest,
    page: Optional[int] = Query(default=1, description="current page"),
    page_size: Optional[int] = Query(default=10, description="page size"),
    service: Service = Depends(get_service),
) -> Result[PaginationResult[ServerResponse]]:
    """Query Conversation entities

    Args:
        request (ServeRequest): The request
        page (int): The page number
        page_size (int): The page size
        service (Service): The service
    Returns:
        ServerResponse: The response
    """
    # 调用服务对象的分页查询方法，获取符合条件的会话实体列表，并返回成功的结果
    return Result.succ(service.get_list_by_page(request, page, page_size))


@router.get(
    "/list",
    response_model=Result[List[ServerResponse]],
    dependencies=[Depends(check_api_key)],
)
async def list_latest_conv(
    user_name: str = None,
    user_id: str = None,
    sys_code: str = None,
    page: Optional[int] = Query(default=1, description="current page"),
    page_size: Optional[int] = Query(default=10, description="page size"),
    service: Service = Depends(get_service),
) -> Result[List[ServerResponse]]:
    """Return latest conversations"""
    # 创建一个服务请求对象，使用提供的用户信息和系统代码
    request = ServeRequest(
        user_name=user_name or user_id,
        sys_code=sys_code,
    )
    # 调用服务对象的分页查询方法，获取最新会话列表，并返回成功的结果
    return Result.succ(service.get_list_by_page(request, page, page_size).items)


@router.get(
    "/messages/history",
    response_model=Result[List[MessageVo]],
    dependencies=[Depends(check_api_key)],
)
async def get_history_messages(con_uid: str, service: Service = Depends(get_service)):
    """Get the history messages of a conversation"""
    # 调用服务对象的获取历史消息方法，获取指定会话UID的历史消息，并返回成功的结果
    return Result.succ(service.get_history_messages(ServeRequest(conv_uid=con_uid)))


def init_endpoints(system_app: SystemApp) -> None:
    """Initialize the endpoints"""
    # 将服务类注册到系统应用中
    system_app.register(Service)
    # 将全局系统应用设为当前系统应用
    global global_system_app
    global_system_app = system_app
```