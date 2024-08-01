# `.\DB-GPT-src\dbgpt\util\fastapi.py`

```py
"""FastAPI utilities."""

import importlib.metadata as metadata
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI
from fastapi.routing import APIRouter

# 获取 FastAPI 的版本信息
_FASTAPI_VERSION = metadata.version("fastapi")

class PriorityAPIRouter(APIRouter):
    """A router with priority.

    The route with higher priority will be put in the front of the route list.
    """

    def __init__(self, *args, **kwargs):
        """Init a PriorityAPIRouter."""
        super().__init__(*args, **kwargs)
        # 创建存储路由优先级的字典
        self.route_priority: Dict[str, int] = {}

    def add_api_route(
        self, path: str, endpoint: Callable, *, priority: int = 0, **kwargs: Any
    ):
        """Add a route with priority.

        Args:
            path (str): The path of the route.
            endpoint (Callable): The endpoint of the route.
            priority (int, optional): The priority of the route. Defaults to 0.
            **kwargs (Any): Other arguments.
        """
        # 调用父类方法添加路由
        super().add_api_route(path, endpoint, **kwargs)
        # 将路由优先级信息存储到字典中
        self.route_priority[path] = priority
        # 根据优先级对路由进行排序
        self.sort_routes_by_priority()

    def sort_routes_by_priority(self):
        """Sort the routes by priority."""

        def my_func(route):
            if route.path in ["", "/"]:
                return -100
            return self.route_priority.get(route.path, 0)

        # 根据自定义的排序函数对路由进行排序（降序）
        self.routes.sort(key=my_func, reverse=True)


_HAS_STARTUP = False
_HAS_SHUTDOWN = False
_GLOBAL_STARTUP_HANDLERS: List[Callable] = []

_GLOBAL_SHUTDOWN_HANDLERS: List[Callable] = []


def register_event_handler(app: FastAPI, event: str, handler: Callable):
    """Register an event handler.

    Args:
        app (FastAPI): The FastAPI app.
        event (str): The event type.
        handler (Callable): The handler function.

    """
    if _FASTAPI_VERSION >= "0.109.1":
        # 检查 FastAPI 版本是否支持全局事件处理器
        if event == "startup":
            # 添加启动事件处理器到全局列表
            if _HAS_STARTUP:
                raise ValueError(
                    "FastAPI app already started. Cannot add startup handler."
                )
            _GLOBAL_STARTUP_HANDLERS.append(handler)
        elif event == "shutdown":
            # 添加关闭事件处理器到全局列表
            if _HAS_SHUTDOWN:
                raise ValueError(
                    "FastAPI app already shutdown. Cannot add shutdown handler."
                )
            _GLOBAL_SHUTDOWN_HANDLERS.append(handler)
        else:
            # 抛出异常，事件类型无效
            raise ValueError(f"Invalid event: {event}")
    else:
        # 使用旧版本 FastAPI，直接添加事件处理器到 FastAPI 应用
        if event == "startup":
            app.add_event_handler("startup", handler)
        elif event == "shutdown":
            app.add_event_handler("shutdown", handler)
        else:
            # 抛出异常，事件类型无效
            raise ValueError(f"Invalid event: {event}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 触发启动事件
    global _HAS_STARTUP, _HAS_SHUTDOWN
    for handler in _GLOBAL_STARTUP_HANDLERS:
        await handler()
    # 设置全局变量 _HAS_STARTUP 为 True，表示程序启动已经发生
    _HAS_STARTUP = True
    # 使用生成器函数的 yield 关键字，暂停函数执行并返回值
    yield
    # 触发全局的关闭事件
    for handler in _GLOBAL_SHUTDOWN_HANDLERS:
        # 等待每个处理程序完成其任务
        await handler()
    # 设置全局变量 _HAS_SHUTDOWN 为 True，表示程序已经执行了关闭操作
    _HAS_SHUTDOWN = True
# 创建一个 FastAPI 应用程序
def create_app(*args, **kwargs) -> FastAPI:
    # 初始化生命周期变量为 None
    _sp = None
    # 如果 FastAPI 版本大于等于 "0.109.1"
    if _FASTAPI_VERSION >= "0.109.1":
        # 如果关键字参数中没有指定 "lifespan"，则使用默认的生命周期设置
        if "lifespan" not in kwargs:
            kwargs["lifespan"] = lifespan
        # 将生命周期设置存储在 _sp 变量中
        _sp = kwargs["lifespan"]
    # 创建 FastAPI 应用程序实例
    app = FastAPI(*args, **kwargs)
    # 如果存在自定义的生命周期设置，则将其存储在 app 对象的特定属性中
    if _sp:
        app.__dbgpt_custom_lifespan = _sp
    return app


# 替换 FastAPI 应用程序的路由器
def replace_router(app: FastAPI, router: Optional[APIRouter] = None):
    # 如果未提供 router 参数，则使用 PriorityAPIRouter 创建一个默认路由器
    if not router:
        router = PriorityAPIRouter()
    # 如果 FastAPI 版本大于等于 "0.109.1"
    if _FASTAPI_VERSION >= "0.109.1":
        # 如果应用程序对象具有 "__dbgpt_custom_lifespan" 属性
        if hasattr(app, "__dbgpt_custom_lifespan"):
            # 获取存储在 app 对象中的自定义生命周期设置
            _sp = getattr(app, "__dbgpt_custom_lifespan")
            # 将自定义生命周期设置应用于新的路由器对象
            router.lifespan_context = _sp

    # 将新的路由器对象赋值给应用程序的 router 属性
    app.router = router
    # 执行应用程序的 setup 方法，进行初始化设置
    app.setup()
    return app
```