# `.\DB-GPT-src\dbgpt\serve\prompt\tests\test_endpoints.py`

```py
# 导入 pytest 库，用于编写和运行测试用例
import pytest
# 导入 FastAPI 类，用于创建 FastAPI 应用程序
from fastapi import FastAPI
# 导入 AsyncClient 类，用于异步测试客户端
from httpx import AsyncClient

# 导入 SystemApp 类，来自 dbgpt.component 模块
from dbgpt.component import SystemApp
# 导入 asystem_app 和 client，来自 dbgpt.serve.core.tests.conftest 模块
from dbgpt.serve.core.tests.conftest import asystem_app, client
# 导入 db 对象，来自 dbgpt.storage.metadata 模块，用于数据库操作
from dbgpt.storage.metadata import db
# 导入 PaginationResult 类，用于分页结果处理，来自 dbgpt.util 模块
from dbgpt.util import PaginationResult

# 导入 init_endpoints 和 router，来自 ..api.endpoints 模块
from ..api.endpoints import init_endpoints, router
# 导入 ServerResponse 类，用于处理服务器响应，来自 ..api.schemas 模块
from ..api.schemas import ServerResponse
# 导入 SERVE_CONFIG_KEY_PREFIX 常量，用于配置前缀信息，来自 ..config 模块
from ..config import SERVE_CONFIG_KEY_PREFIX

# 定义 fixture，初始化数据库并在测试后销毁
@pytest.fixture(autouse=True)
def setup_and_teardown():
    db.init_db("sqlite:///:memory:")  # 使用 SQLite 内存数据库初始化
    db.create_all()  # 创建数据库表结构

    yield  # 返回测试执行的控制权

# 初始化客户端调用函数，将路由器包含到 FastAPI 应用程序中，并初始化端点
def client_init_caller(app: FastAPI, system_app: SystemApp):
    app.include_router(router)  # 将路由器包含到 FastAPI 应用程序中
    init_endpoints(system_app)  # 初始化端点

# 异步函数，创建并验证操作
async def _create_and_validate(
    client: AsyncClient, sys_code: str, content: str, expect_id: int = 1, **kwargs
):
    req_json = {"sys_code": sys_code, "content": content}  # 构建请求 JSON 数据
    req_json.update(kwargs)  # 更新请求 JSON 数据
    response = await client.post("/add", json=req_json)  # 发送 POST 请求到指定路径
    assert response.status_code == 200  # 断言响应状态码为 200
    json_res = response.json()  # 将响应转换为 JSON 格式
    assert "success" in json_res and json_res["success"]  # 断言响应中包含成功标志且为真
    assert "data" in json_res and json_res["data"]  # 断言响应中包含数据字段
    data = json_res["data"]  # 获取数据字段内容
    res_obj = ServerResponse(**data)  # 构建 ServerResponse 对象
    assert res_obj.id == expect_id  # 断言返回的对象 ID 符合预期
    assert res_obj.sys_code == sys_code  # 断言返回的对象系统代码符合预期
    assert res_obj.content == content  # 断言返回的对象内容符合预期

# 异步测试函数，测试 API 健康检查功能
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client, asystem_app, has_auth",
    [
        (
            {
                "app_caller": client_init_caller,  # 设置客户端调用函数
                "client_api_key": "test_token1",  # 设置客户端 API 密钥
            },
            {
                "app_config": {
                    f"{SERVE_CONFIG_KEY_PREFIX}api_keys": "test_token1,test_token2"
                }  # 设置应用配置信息
            },
            True,  # 设置验证是否通过
        ),
        (
            {
                "app_caller": client_init_caller,  # 设置客户端调用函数
                "client_api_key": "error_token",  # 设置错误的客户端 API 密钥
            },
            {
                "app_config": {
                    f"{SERVE_CONFIG_KEY_PREFIX}api_keys": "test_token1,test_token2"
                }  # 设置应用配置信息
            },
            False,  # 设置验证未通过
        ),
    ],
    indirect=["client", "asystem_app"],  # 指定间接参数化的依赖项
)
async def test_api_health(client: AsyncClient, asystem_app, has_auth: bool):
    response = await client.get("/test_auth")  # 发送 GET 请求到 /test_auth 路径
    if has_auth:
        assert response.status_code == 200  # 断言响应状态码为 200
        assert response.json() == {"status": "ok"}  # 断言返回的 JSON 数据为 {"status": "ok"}

# 异步测试函数，测试 API 授权功能
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]  # 客户端参数化间接依赖
)
async def test_api_auth(client: AsyncClient):
    response = await client.get("/health")  # 发送 GET 请求到 /health 路径
    response.raise_for_status()  # 抛出异常以处理错误响应
    assert response.status_code == 200  # 断言响应状态码为 200
    assert response.json() == {"status": "ok"}  # 断言返回的 JSON 数据为 {"status": "ok"}

# 异步测试函数，测试创建 API 功能
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]  # 客户端参数化间接依赖
)
async def test_api_create(client: AsyncClient):
    await _create_and_validate(client, "test", "test")  # 调用 _create_and_validate 函数进行测试

# 定义未完成的测试用例，等待补充
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]  # 客户端参数化间接依赖
)
async def test_api_update(client: AsyncClient):
    # 创建并验证测试数据
    await _create_and_validate(client, "test", "test")

    # 发送 POST 请求更新数据
    response = await client.post("/update", json={"id": 1, "content": "test2"})
    # 断言响应状态码为 200
    assert response.status_code == 200
    # 解析 JSON 响应体
    json_res = response.json()
    # 断言返回的 JSON 中包含 "success" 并且其值为 True
    assert "success" in json_res and json_res["success"]
    # 断言返回的 JSON 中包含 "data"
    assert "data" in json_res and json_res["data"]
    # 获取 "data" 字段的值
    data = json_res["data"]
    # 将 "data" 转换为 ServerResponse 对象
    res_obj = ServerResponse(**data)
    # 断言 ServerResponse 对象的属性值符合预期
    assert res_obj.id == 1
    assert res_obj.sys_code == "test"
    assert res_obj.content == "test2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]
)
async def test_api_query(client: AsyncClient):
    # 创建并验证多个测试数据
    for i in range(10):
        await _create_and_validate(
            client, "test", f"test{i}", expect_id=i + 1, prompt_name=f"prompt_name_{i}"
        )
    # 发送 POST 请求查询数据列表
    response = await client.post("/list", json={"sys_code": "test"})
    # 断言响应状态码为 200
    assert response.status_code == 200
    # 解析 JSON 响应体
    json_res = response.json()
    # 断言返回的 JSON 中包含 "success" 并且其值为 True
    assert "success" in json_res and json_res["success"]
    # 断言返回的 JSON 中包含 "data"
    assert "data" in json_res and json_res["data"]
    # 获取 "data" 字段的值
    data = json_res["data"]
    # 断言返回的数据列表长度为 10
    assert len(data) == 10
    # 将第一个数据项转换为 ServerResponse 对象
    res_obj = ServerResponse(**data[0])
    # 断言 ServerResponse 对象的属性值符合预期
    assert res_obj.id == 1
    assert res_obj.sys_code == "test"
    assert res_obj.content == "test0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client", [{"app_caller": client_init_caller}], indirect=["client"]
)
async def test_api_query_by_page(client: AsyncClient):
    # 创建并验证多个测试数据
    for i in range(10):
        await _create_and_validate(
            client, "test", f"test{i}", expect_id=i + 1, prompt_name=f"prompt_name_{i}"
        )
    # 发送带分页参数的 POST 请求查询数据页
    response = await client.post(
        "/query_page", params={"page": 1, "page_size": 5}, json={"sys_code": "test"}
    )
    # 断言响应状态码为 200
    assert response.status_code == 200
    # 解析 JSON 响应体
    json_res = response.json()
    # 断言返回的 JSON 中包含 "success" 并且其值为 True
    assert "success" in json_res and json_res["success"]
    # 断言返回的 JSON 中包含 "data"
    assert "data" in json_res and json_res["data"]
    # 获取 "data" 字段的值
    data = json_res["data"]
    # 将返回的分页结果转换为 PaginationResult 对象
    page_result: PaginationResult = PaginationResult(**data)
    # 断言分页结果的总条目数为 10
    assert page_result.total_count == 10
    # 断言分页结果的总页数为 2
    assert page_result.total_pages == 2
    # 断言分页结果的当前页为 1
    assert page_result.page == 1
    # 断言分页结果的每页大小为 5
    assert page_result.page_size == 5
    # 断言分页结果中条目的数量为 5
    assert len(page_result.items) == 5
```