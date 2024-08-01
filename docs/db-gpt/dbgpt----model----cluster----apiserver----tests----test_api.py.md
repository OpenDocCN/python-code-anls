# `.\DB-GPT-src\dbgpt\model\cluster\apiserver\tests\test_api.py`

```py
import pytest  # 导入 pytest 库，用于编写和运行测试用例
import pytest_asyncio  # 导入 pytest_asyncio 库，支持 asyncio 的 pytest 插件
from fastapi.middleware.cors import CORSMiddleware  # 导入 FastAPI 的 CORS 中间件
from httpx import ASGITransport, AsyncClient, HTTPError  # 导入 httpx 库中的异步 HTTP 客户端相关模块

from dbgpt.component import SystemApp  # 导入自定义模块 SystemApp
from dbgpt.model.cluster.apiserver.api import (  # 导入自定义模块中的一组类和函数
    ModelList,
    api_settings,
    initialize_apiserver,
)
from dbgpt.model.cluster.tests.conftest import _new_cluster  # 导入测试配置模块中的函数
from dbgpt.model.cluster.worker.manager import _DefaultWorkerManagerFactory  # 导入工作管理相关模块
from dbgpt.util.fastapi import create_app  # 导入自定义模块中的 FastAPI 应用创建函数
from dbgpt.util.openai_utils import chat_completion, chat_completion_stream  # 导入自定义模块中的 OpenAI 相关函数

app = create_app()  # 创建 FastAPI 应用实例

app.add_middleware(  # 添加 CORS 中间件到 FastAPI 应用
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@pytest_asyncio.fixture  # 定义一个 asyncio 的 pytest fixture，返回 SystemApp 对象
async def system_app():
    return SystemApp(app)


@pytest_asyncio.fixture  # 定义一个 asyncio 的 pytest fixture，返回 AsyncClient 对象
async def client(request, system_app: SystemApp):
    param = getattr(request, "param", {})  # 从请求中获取参数字典
    api_keys = param.get("api_keys", [])  # 获取参数字典中的 api_keys，若不存在则返回空列表
    client_api_key = param.get("client_api_key")  # 获取参数字典中的 client_api_key
    if "num_workers" not in param:
        param["num_workers"] = 2  # 如果参数字典中没有 num_workers 键，则设置为默认值 2
    if "api_keys" in param:
        del param["api_keys"]  # 如果参数字典中有 api_keys 键，则删除它
    headers = {}  # 初始化 headers 字典
    if client_api_key:
        headers["Authorization"] = "Bearer " + client_api_key  # 如果有 client_api_key，则设置 Authorization 头部
    print(f"param: {param}")  # 打印参数信息
    if api_settings:
        # 清空全局 api keys
        api_settings.api_keys = []
    async with AsyncClient(  # 使用 httpx 的异步客户端，设置传输方式为 ASGI，基础 URL 为 "http://test"，设置 headers
        transport=ASGITransport(app), base_url="http://test", headers=headers
    ) as client:
        async with _new_cluster(**param) as cluster:  # 使用 _new_cluster 函数创建集群
            worker_manager, model_registry = cluster  # 解包集群返回的 worker_manager 和 model_registry
            system_app.register(_DefaultWorkerManagerFactory, worker_manager)  # 将 worker_manager 注册到 SystemApp
            system_app.register_instance(model_registry)  # 注册 model_registry 到 SystemApp
            initialize_apiserver(None, None, app, system_app, api_keys=api_keys)  # 初始化 API 服务器
            yield client  # 返回客户端对象


@pytest.mark.asyncio  # 标记异步测试用例
async def test_get_all_models(client: AsyncClient):
    res = await client.get("/api/v1/models")  # 发送 GET 请求获取所有模型信息
    res.status_code == 200  # 检查响应状态码是否为 200
    model_lists = ModelList.model_validate(res.json())  # 验证并解析响应的 JSON 数据
    print(f"model list json: {res.json()}")  # 打印模型列表的 JSON 数据
    assert model_lists.object == "list"  # 断言模型列表的 object 属性为 "list"
    assert len(model_lists.data) == 2  # 断言模型列表中的数据长度为 2


@pytest.mark.asyncio  # 标记异步测试用例
@pytest.mark.parametrize(  # 参数化测试用例，指定不同的参数组合
    "client, expected_messages",
    [
        ({"stream_messags": ["Hello", " world."]}, "Hello world."),  # 第一组参数
        ({"stream_messags": ["你好，我是", "张三。"]}, "你好，我是张三。"),  # 第二组参数
    ],
    indirect=["client"],  # 指定 client 参数为间接参数
)
async def test_chat_completions(client: AsyncClient, expected_messages):
    chat_data = {  # 定义聊天数据字典
        "model": "test-model-name-0",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": True,
    }
    full_text = ""  # 初始化完整文本为空字符串
    async for text in chat_completion_stream(  # 使用异步 for 循环处理聊天完成流
        "/api/v1/chat/completions", chat_data, client
    ):
        full_text += text  # 将每次迭代的文本添加到完整文本中
    assert full_text == expected_messages  # 断言完整文本与预期消息一致

    assert (  # 断言单个聊天完成请求的结果与预期消息一致
        await chat_completion("/api/v1/chat/completions", chat_data, client)
        == expected_messages
    )


@pytest.mark.asyncio  # 标记异步测试用例
@pytest.mark.parametrize(
    "client, expected_messages, client_api_key",
    [
        (
            {"stream_messags": ["Hello", " world."], "api_keys": ["abc"]},
            "Hello world.",
            "abc",
        ),
        (
            {"stream_messags": ["你好，我是", "张三。"], "api_keys": ["abc"]},
            "你好，我是张三。",
            "abc",
        ),
    ],
    indirect=["client"],
)
async def test_chat_completions_with_openai_lib_async_no_stream(
    client: AsyncClient, expected_messages: str, client_api_key: str
):
    """
    使用参数化测试来测试异步函数 `test_chat_completions_with_openai_lib_async_no_stream`，
    测试不涉及流式处理。

    Args:
        client (AsyncClient): 异步 HTTP 客户端对象
        expected_messages (str): 预期的聊天消息
        client_api_key (str): 客户端的 API 密钥
    """
    # import openai
    #
    # openai.api_key = client_api_key
    # openai.api_base = "http://test/api/v1"
    #
    # model_name = "test-model-name-0"
    #
    # with aioresponses() as mocked:
    #     mock_message = {"text": expected_messages}
    #     one_res = ChatCompletionResponseChoice(
    #         index=0,
    #         message=ChatMessage(role="assistant", content=expected_messages),
    #         finish_reason="stop",
    #     )
    #     data = ChatCompletionResponse(
    #         model=model_name, choices=[one_res], usage=UsageInfo()
    #     )
    #     mock_message = f"{data.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    #     # Mock http request
    #     mocked.post(
    #         "http://test/api/v1/chat/completions", status=200, body=mock_message
    #     )
    #     completion = await openai.ChatCompletion.acreate(
    #         model=model_name,
    #         messages=[{"role": "user", "content": "Hello! What is your name?"}],
    #     )
    #     assert completion.choices[0].message.content == expected_messages
    # TODO test openai lib
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client, expected_messages, client_api_key",
    [
        (
            {"stream_messags": ["Hello", " world."], "api_keys": ["abc"]},
            "Hello world.",
            "abc",
        ),
        (
            {"stream_messags": ["你好，我是", "张三。"], "api_keys": ["abc"]},
            "你好，我是张三。",
            "abc",
        ),
    ],
    indirect=["client"],
)
async def test_chat_completions_with_openai_lib_async_stream(
    client: AsyncClient, expected_messages: str, client_api_key: str
):
    """
    使用参数化测试来测试异步函数 `test_chat_completions_with_openai_lib_async_stream`，
    测试涉及流式处理。

    Args:
        client (AsyncClient): 异步 HTTP 客户端对象
        expected_messages (str): 预期的聊天消息
        client_api_key (str): 客户端的 API 密钥
    """
    # import openai
    #
    # openai.api_key = client_api_key
    # openai.api_base = "http://test/api/v1"
    #
    # model_name = "test-model-name-0"
    #
    # with aioresponses() as mocked:
    #     mock_message = {"text": expected_messages}
    #     choice_data = ChatCompletionResponseStreamChoice(
    #         index=0,
    #         delta=DeltaMessage(content=expected_messages),
    #         finish_reason="stop",
    #     )
    #     chunk = ChatCompletionStreamResponse(
    #         id=0, choices=[choice_data], model=model_name
    #     )
    #     mock_message = f"data: {chunk.json(exclude_unset=True, ensure_ascii=False)}\n\n"
    #     mocked.post(
    #         "http://test/api/v1/chat/completions",
    #         status=200,
    #         body=mock_message,
    #         headers={"Content-Type": "text/event-stream"},
    #     )
    #     completion = await openai.ChatCompletion.acreate(
    #         model=model_name,
    #         messages=[{"role": "user", "content": "Hello! What is your name?"}],
    #         stream=True,
    #     )
    #     assert completion.choices[0].delta.content == expected_messages
    # TODO test openai lib
    pass
    # 设置 content_type 为 "text/event-stream"
    #     )
    #
    #     初始化 stream_stream_resp 为空字符串
    #     如果 metadata.version("openai") 大于或等于 "1.0.0":
    #         从 openai 模块导入 OpenAI 类
    #
    #         创建 OpenAI 客户端对象 client，使用给定的 base_url 和 api_key 参数
    #         client = OpenAI(
    #             **{"base_url": "http://test/api/v1", "api_key": client_api_key}
    #         )
    #         调用 client 的 chat.completions.create 方法，发起聊天完成请求：
    #         - 使用指定的 model_name 模型名
    #         - 发送一条用户角色的消息 "Hello! What is your name?"
    #         - 设置 stream 参数为 True，表示开启流式处理
    #         res = await client.chat.completions.create(
    #             model=model_name,
    #             messages=[{"role": "user", "content": "Hello! What is your name?"}],
    #             stream=True,
    #         )
    #     否则，如果版本不符合要求，使用 openai 模块的 ChatCompletion 类的 acreate 方法：
    #         - 使用指定的 model_name 模型名
    #         - 发送一条用户角色的消息 "Hello! What is your name?"
    #         - 设置 stream 参数为 True，表示开启流式处理
    #         res = openai.ChatCompletion.acreate(
    #             model=model_name,
    #             messages=[{"role": "user", "content": "Hello! What is your name?"}],
    #             stream=True,
    #         )
    #     使用异步迭代从 res 中获取 stream_resp 对象，执行以下循环体：
    #         - 从 stream_resp 的 choices 列表中获取第一个选择项
    #         - 获取该选择项中 delta 字典中的 "content" 键对应的值，赋给 stream_stream_resp
    #
    #     断言 stream_stream_resp 等于预期的 expected_messages
    # TODO test openai lib
    pass
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client, expected_messages, api_key_is_error",
    [  # 定义参数化测试的参数列表
        (
            {  # 第一组参数化测试数据
                "stream_messags": ["Hello", " world."],  # 模拟客户端的消息流
                "api_keys": ["abc", "xx"],  # 模拟可用的 API 密钥列表
                "client_api_key": "abc",  # 客户端使用的 API 密钥
            },
            "Hello world.",  # 预期的聊天完成消息
            False,  # 预期此测试不会引发 API 密钥错误
        ),
        (  # 第二组参数化测试数据
            {"stream_messags": ["你好，我是", "张三。"]},  # 模拟中文消息流
            "你好，我是张三。",  # 预期的中文聊天完成消息
            False,  # 预期此测试不会引发 API 密钥错误
        ),
        (  # 第三组参数化测试数据
            {  # 第三组包含 API 密钥的参数化测试数据
                "stream_messags": ["你好，我是", "张三。"],  # 模拟中文消息流
                "api_keys": ["abc", "xx"],  # 模拟可用的 API 密钥列表
            },
            "你好，我是张三。",  # 预期的中文聊天完成消息
            True,  # 预期此测试会引发 API 密钥错误
        ),
        (  # 第四组参数化测试数据
            {  # 第四组包含错误 API 密钥的参数化测试数据
                "stream_messags": ["你好，我是", "张三。"],  # 模拟中文消息流
                "api_keys": ["abc", "xx"],  # 模拟可用的 API 密钥列表
                "client_api_key": "error_api_key",  # 错误的客户端 API 密钥
            },
            "你好，我是张三。",  # 预期的中文聊天完成消息
            True,  # 预期此测试会引发 API 密钥错误
        ),
    ],
    indirect=["client"],  # 指定 'client' 参数为间接参数，由 pytest 根据参数化测试提供
)
async def test_chat_completions_with_api_keys(
    client: AsyncClient, expected_messages: str, api_key_is_error: bool
):
    chat_data = {  # 定义聊天数据
        "model": "test-model-name-0",  # 模型名称
        "messages": [{"role": "user", "content": "Hello"}],  # 用户消息列表
        "stream": True,  # 是否流式传输
    }
    if api_key_is_error:
        with pytest.raises(HTTPError):  # 断言会抛出 HTTPError 异常
            await chat_completion("/api/v1/chat/completions", chat_data, client)  # 调用异步聊天完成函数
    else:
        assert (  # 断言聊天完成函数的返回值与预期消息相符
            await chat_completion("/api/v1/chat/completions", chat_data, client)
            == expected_messages
        )
```