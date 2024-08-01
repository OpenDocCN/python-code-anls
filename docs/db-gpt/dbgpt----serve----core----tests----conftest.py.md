# `.\DB-GPT-src\dbgpt\serve\core\tests\conftest.py`

```py
# 引入必要的模块和类型提示
from typing import Dict

import pytest  # 引入pytest测试框架
import pytest_asyncio  # 引入pytest异步支持
from fastapi.middleware.cors import CORSMiddleware  # 引入FastAPI的跨域中间件
from httpx import ASGITransport, AsyncClient  # 引入httpx的ASGITransport和AsyncClient类

from dbgpt.component import SystemApp  # 引入自定义模块中的SystemApp类
from dbgpt.util import AppConfig  # 引入自定义模块中的AppConfig类
from dbgpt.util.fastapi import create_app  # 引入自定义模块中的create_app函数


# 创建系统应用的函数，接受一个字典参数并返回SystemApp对象
def create_system_app(param: Dict) -> SystemApp:
    # 从参数中获取应用配置
    app_config = param.get("app_config", {})
    # 如果app_config是字典，则创建一个AppConfig对象
    if isinstance(app_config, dict):
        app_config = AppConfig(configs=app_config)
    # 如果app_config不是AppConfig类型，抛出异常
    elif not isinstance(app_config, AppConfig):
        raise RuntimeError("app_config must be AppConfig or dict")

    # 创建FastAPI应用实例
    test_app = create_app()
    # 添加跨域中间件到应用
    test_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # 返回SystemApp对象，将FastAPI应用和配置对象传入
    return SystemApp(test_app, app_config)


# 异步fixture，用于创建系统应用对象
@pytest_asyncio.fixture
async def asystem_app(request):
    param = getattr(request, "param", {})
    return create_system_app(param)


# fixture，用于创建系统应用对象
@pytest.fixture
def system_app(request):
    param = getattr(request, "param", {})
    return create_system_app(param)


# 异步fixture，用于创建带有HTTP客户端的测试客户端对象
@pytest_asyncio.fixture
async def client(request, asystem_app: SystemApp):
    param = getattr(request, "param", {})
    headers = param.get("headers", {})  # 获取请求头部信息
    base_url = param.get("base_url", "http://test")  # 获取基础URL，默认为"http://test"
    client_api_key = param.get("client_api_key")  # 获取客户端API密钥
    routers = param.get("routers", [])  # 获取路由列表，默认为空列表
    app_caller = param.get("app_caller")  # 获取应用调用者函数

    # 如果参数中存在"api_keys"键，则删除该键
    if "api_keys" in param:
        del param["api_keys"]
    # 如果客户端API密钥存在，则将其添加到请求头部的Authorization字段中
    if client_api_key:
        headers["Authorization"] = "Bearer " + client_api_key

    # 获取asystem_app中的FastAPI应用实例
    test_app = asystem_app.app

    # 使用AsyncClient创建异步HTTP客户端
    async with AsyncClient(
        transport=ASGITransport(test_app), base_url=base_url, headers=headers
    ) as client:
        # 将传入的路由添加到FastAPI应用中
        for router in routers:
            test_app.include_router(router)
        # 如果存在应用调用者函数，则调用该函数，传入应用和asystem_app
        if app_caller:
            app_caller(test_app, asystem_app)
        # 返回HTTP客户端对象
        yield client
```