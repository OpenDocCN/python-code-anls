# `.\Langchain-Chatchat\server\utils.py`

```py
# 导入必要的库
import pydantic
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from pathlib import Path
import asyncio
# 从configs模块中导入所需的变量和函数
from configs import (LLM_MODELS, LLM_DEVICE, EMBEDDING_DEVICE,
                     MODEL_PATH, MODEL_ROOT_PATH, ONLINE_LLM_MODEL, logger, log_verbose,
                     FSCHAT_MODEL_WORKERS, HTTPX_DEFAULT_TIMEOUT)
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
# 从langchain模块中导入ChatOpenAI和OpenAI类
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import httpx
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Callable,
    Generator,
    Dict,
    Any,
    Awaitable,
    Union,
    Tuple
)
import logging
import torch

# 从minx_chat_openai模块中导入MinxChatOpenAI类
from server.minx_chat_openai import MinxChatOpenAI

# 定义一个异步函数，用于包装一个可等待对象，并在完成或引发异常时发出信号
async def wrap_done(fn: Awaitable, event: asyncio.Event):
    """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
    try:
        await fn
    except Exception as e:
        logging.exception(e)
        msg = f"Caught exception: {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
    finally:
        # Signal the aiter to stop.
        event.set()

# 定义一个函数，用于获取ChatOpenAI对象
def get_ChatOpenAI(
        model_name: str,
        temperature: float,
        max_tokens: int = None,
        streaming: bool = True,
        callbacks: List[Callable] = [],
        verbose: bool = True,
        **kwargs: Any,
) -> ChatOpenAI:
    # 获取模型工作配置
    config = get_model_worker_config(model_name)
    # 如果模型名称为"openai-api"，则从配置中获取实际模型名称
    if model_name == "openai-api":
        model_name = config.get("model_name")
    # 设置ChatOpenAI类的_get_encoding_model方法为MinxChatOpenAI类的get_encoding_model方法
    ChatOpenAI._get_encoding_model = MinxChatOpenAI.get_encoding_model
    # 创建一个 ChatOpenAI 对象，传入参数 streaming、verbose、callbacks、openai_api_key、openai_api_base、model_name、temperature、max_tokens、openai_proxy 和 kwargs
    model = ChatOpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),  # 获取配置文件中的 API 密钥，如果没有则使用默认值 "EMPTY"
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),  # 获取配置文件中的 API 基础 URL，如果没有则使用默认值 fschat_openai_api_address() 的返回值
        model_name=model_name,  # 模型名称
        temperature=temperature,  # 温度参数
        max_tokens=max_tokens,  # 最大 token 数
        openai_proxy=config.get("openai_proxy"),  # 获取配置文件中的 OpenAI 代理信息
        **kwargs  # 其他关键字参数
    )
    # 返回创建的 ChatOpenAI 对象
    return model
# 定义一个函数，用于创建并返回一个 OpenAI 对象
def get_OpenAI(
        model_name: str,  # 模型名称
        temperature: float,  # 温度参数
        max_tokens: int = None,  # 最大 token 数量，默认为 None
        streaming: bool = True,  # 是否启用流式生成，默认为 True
        echo: bool = True,  # 是否回显输入，默认为 True
        callbacks: List[Callable] = [],  # 回调函数列表，默认为空列表
        verbose: bool = True,  # 是否启用详细输出，默认为 True
        **kwargs: Any,  # 其他参数
) -> OpenAI:  # 返回类型为 OpenAI 类型
    # 获取模型的配置信息
    config = get_model_worker_config(model_name)
    # 如果模型名称为 "openai-api"，则从配置中获取实际模型名称
    if model_name == "openai-api":
        model_name = config.get("model_name")
    # 创建 OpenAI 对象
    model = OpenAI(
        streaming=streaming,
        verbose=verbose,
        callbacks=callbacks,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        openai_proxy=config.get("openai_proxy"),
        echo=echo,
        **kwargs
    )
    # 返回创建的 OpenAI 对象
    return model


# 定义一个基础响应类，继承自基础模型类
class BaseResponse(BaseModel):
    code: int = pydantic.Field(200, description="API status code")  # 状态码，默认为 200
    msg: str = pydantic.Field("success", description="API status message")  # 状态消息，默认为 "success"
    data: Any = pydantic.Field(None, description="API data")  # API 数据，默认为 None

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
            }
        }


# 定义一个列表响应类，继承自基础响应类
class ListResponse(BaseResponse):
    data: List[str] = pydantic.Field(..., description="List of names")  # 名称列表

    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "msg": "success",
                "data": ["doc1.docx", "doc2.pdf", "doc3.txt"],
            }
        }


# 定义一个聊天消息类，继承自基础模型类
class ChatMessage(BaseModel):
    question: str = pydantic.Field(..., description="Question text")  # 问题文本
    response: str = pydantic.Field(..., description="Response text")  # 回复文本
    history: List[List[str]] = pydantic.Field(..., description="History text")  # 历史文本
    source_documents: List[str] = pydantic.Field(
        ..., description="List of source documents and their scores"
    )  # 源文档列表及其分数
    # 定义一个 Config 类
    class Config:
        # 定义 schema_extra 属性，包含示例数据
        schema_extra = {
            "example": {
                # 示例数据中包含问题和回答的信息
                "question": "工伤保险如何办理？",
                "response": "根据已知信息，可以总结如下：\n\n1. 参保单位为员工缴纳工伤保险费，以保障员工在发生工伤时能够获得相应的待遇。\n"
                            "2. 不同地区的工伤保险缴费规定可能有所不同，需要向当地社保部门咨询以了解具体的缴费标准和规定。\n"
                            "3. 工伤从业人员及其近亲属需要申请工伤认定，确认享受的待遇资格，并按时缴纳工伤保险费。\n"
                            "4. 工伤保险待遇包括工伤医疗、康复、辅助器具配置费用、伤残待遇、工亡待遇、一次性工亡补助金等。\n"
                            "5. 工伤保险待遇领取资格认证包括长期待遇领取人员认证和一次性待遇领取人员认证。\n"
                            "6. 工伤保险基金支付的待遇项目包括工伤医疗待遇、康复待遇、辅助器具配置费用、一次性工亡补助金、丧葬补助金等。",
                # 示例数据中包含历史问题和回答的信息
                "history": [
                    [
                        "工伤保险是什么？",
                        "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，"
                        "由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                    ]
                ],
                # 示例数据中包含来源文档的信息
                "source_documents": [
                    "出处 [1] 广州市单位从业的特定人员参加工伤保险办事指引.docx：\n\n\t"
                    "( 一)  从业单位  (组织)  按“自愿参保”原则，  为未建 立劳动关系的特定从业人员单项参加工伤保险 、缴纳工伤保 险费。",
                    "出处 [2] ...",
                    "出处 [3] ...",
                ],
            }
        }
# 定义一个函数用于执行 torch 的垃圾回收操作
def torch_gc():
    try:
        # 尝试导入 torch 库
        import torch
        # 如果 CUDA 可用
        if torch.cuda.is_available():
            # 清空 CUDA 缓存
            torch.cuda.empty_cache()
            # 收集 CUDA IPC 内存
            torch.cuda.ipc_collect()
        # 如果使用了 MPS 后端
        elif torch.backends.mps.is_available():
            try:
                # 尝试从 torch.mps 导入 empty_cache 函数
                from torch.mps import empty_cache
                # 调用 empty_cache 函数
                empty_cache()
            except Exception as e:
                # 如果出现异常，记录错误信息
                msg = ("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，"
                       "以支持及时清理 torch 产生的内存占用。")
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
    except Exception:
        ...


# 定义一个函数用于在同步环境中运行异步代码
def run_async(cor):
    '''
    在同步环境中运行异步代码.
    '''
    try:
        # 尝试获取当前事件循环
        loop = asyncio.get_event_loop()
    except:
        # 如果获取失败，创建一个新的事件循环
        loop = asyncio.new_event_loop()
    # 运行并等待异步代码执行完成
    return loop.run_until_complete(cor)


# 定义一个函数用于将异步生成器封装成同步生成器
def iter_over_async(ait, loop=None):
    '''
    将异步生成器封装成同步生成器.
    '''
    # 获取异步生成器的迭代器
    ait = ait.__aiter__()

    # 定义一个异步函数用于获取下一个元素
    async def get_next():
        try:
            obj = await ait.__anext__()
            return False, obj
        except StopAsyncIteration:
            return True, None

    # 如果未指定事件循环，尝试获取当前事件循环，否则创建一个新的事件循环
    if loop is None:
        try:
            loop = asyncio.get_event_loop()
        except:
            loop = asyncio.new_event_loop()

    # 循环直到异步生成器结束
    while True:
        done, obj = loop.run_until_complete(get_next())
        if done:
            break
        yield obj


# 定义一个函数用于修改 FastAPI 对象，使其不依赖 CDN 来展示文档页面
def MakeFastAPIOffline(
        app: FastAPI,
        static_dir=Path(__file__).parent / "static",
        static_url="/static-offline-docs",
        docs_url: Optional[str] = "/docs",
        redoc_url: Optional[str] = "/redoc",
) -> None:
    """patch the FastAPI obj that doesn't rely on CDN for the documentation page"""
    # 导入所需的模块和函数
    from fastapi import Request
    from fastapi.openapi.docs import (
        get_redoc_html,
        get_swagger_ui_html,
        get_swagger_ui_oauth2_redirect_html,
    )
    from fastapi.staticfiles import StaticFiles
    # 导入 starlette.responses 模块中的 HTMLResponse 类
    from starlette.responses import HTMLResponse

    # 获取应用的 OpenAPI URL 和 Swagger UI OAuth2 重定向 URL
    openapi_url = app.openapi_url
    swagger_ui_oauth2_redirect_url = app.swagger_ui_oauth2_redirect_url

    # 定义一个函数，用于移除指定 URL 的路由
    def remove_route(url: str) -> None:
        '''
        remove original route from app
        '''
        index = None
        # 遍历应用的路由列表，查找要移除的路由
        for i, r in enumerate(app.routes):
            if r.path.lower() == url.lower():
                index = i
                break
        # 如果找到要移除的路由，则从路由列表中删除
        if isinstance(index, int):
            app.routes.pop(index)

    # 设置静态文件挂载
    app.mount(
        static_url,
        StaticFiles(directory=Path(static_dir).as_posix()),
        name="static-offline-docs",
    )

    # 如果文档 URL 不为空，则移除文档 URL 和 Swagger UI OAuth2 重定向 URL 的路由
    if docs_url is not None:
        remove_route(docs_url)
        remove_route(swagger_ui_oauth2_redirect_url)

        # 定义文档和 Redoc 页面，指向正确的文件
        @app.get(docs_url, include_in_schema=False)
        async def custom_swagger_ui_html(request: Request) -> HTMLResponse:
            root = request.scope.get("root_path")
            favicon = f"{root}{static_url}/favicon.png"
            return get_swagger_ui_html(
                openapi_url=f"{root}{openapi_url}",
                title=app.title + " - Swagger UI",
                oauth2_redirect_url=swagger_ui_oauth2_redirect_url,
                swagger_js_url=f"{root}{static_url}/swagger-ui-bundle.js",
                swagger_css_url=f"{root}{static_url}/swagger-ui.css",
                swagger_favicon_url=favicon,
            )

        @app.get(swagger_ui_oauth2_redirect_url, include_in_schema=False)
        async def swagger_ui_redirect() -> HTMLResponse:
            return get_swagger_ui_oauth2_redirect_html()
    # 如果 redoc_url 不为空，则执行以下操作
    if redoc_url is not None:
        # 移除已存在的 redoc_url 路由
        remove_route(redoc_url)

        # 创建一个 GET 请求处理函数，用于返回 Redoc 文档的 HTML 页面
        @app.get(redoc_url, include_in_schema=False)
        async def redoc_html(request: Request) -> HTMLResponse:
            # 获取请求的根路径
            root = request.scope.get("root_path")
            # 构建 favicon 的 URL
            favicon = f"{root}{static_url}/favicon.png"

            # 返回 Redoc HTML 页面
            return get_redoc_html(
                openapi_url=f"{root}{openapi_url}",  # OpenAPI 文档的 URL
                title=app.title + " - ReDoc",  # 页面标题
                redoc_js_url=f"{root}{static_url}/redoc.standalone.js",  # Redoc JS 文件的 URL
                with_google_fonts=False,  # 禁用 Google 字体
                redoc_favicon_url=favicon,  # favicon 的 URL
            )
# 从model_config中获取模型信息

# 获取已配置的嵌入模型的名称列表
def list_embed_models() -> List[str]:
    '''
    get names of configured embedding models
    '''
    return list(MODEL_PATH["embed_model"])


# 获取已配置的llm模型信息，包括不同类型的模型
def list_config_llm_models() -> Dict[str, Dict]:
    '''
    get configured llm models with different types.
    return {config_type: {model_name: config}, ...}
    '''
    # 复制FSCHAT_MODEL_WORKERS，去除"default"键
    workers = FSCHAT_MODEL_WORKERS.copy()
    workers.pop("default", None)

    return {
        "local": MODEL_PATH["llm_model"].copy(),
        "online": ONLINE_LLM_MODEL.copy(),
        "worker": workers,
    }


# 获取模型的路径信息
def get_model_path(model_name: str, type: str = None) -> Optional[str]:
    # 如果指定类型在MODEL_PATH中存在，则使用该类型的路径
    if type in MODEL_PATH:
        paths = MODEL_PATH[type]
    else:
        paths = {}
        # 遍历所有MODEL_PATH的值，合并为paths字典
        for v in MODEL_PATH.values():
            paths.update(v)

    # 获取模型名称对应的路径
    if path_str := paths.get(model_name):
        path = Path(path_str)
        # 如果路径为目录，则直接返回
        if path.is_dir():
            return str(path)

        root_path = Path(MODEL_ROOT_PATH)
        if root_path.is_dir():
            path = root_path / model_name
            if path.is_dir():
                return str(path)
            path = root_path / path_str
            if path.is_dir():
                return str(path)
            path = root_path / path_str.split("/")[-1]
            if path.is_dir():
                return str(path)
        return path_str


# 从server_config中获取服务信息

# 获取模型工作配置项
def get_model_worker_config(model_name: str = None) -> dict:
    '''
    加载model worker的配置项。
    优先级:FSCHAT_MODEL_WORKERS[model_name] > ONLINE_LLM_MODEL[model_name] > FSCHAT_MODEL_WORKERS["default"]
    '''
    # 导入必要的模块和变量
    from configs.model_config import ONLINE_LLM_MODEL, MODEL_PATH
    from configs.server_config import FSCHAT_MODEL_WORKERS
    from server import model_workers
    # 从FSCHAT_MODEL_WORKERS字典中获取"default"对应的值，并创建副本
    config = FSCHAT_MODEL_WORKERS.get("default", {}).copy()
    # 更新config字典，将ONLINE_LLM_MODEL中model_name对应的值合并到config中
    config.update(ONLINE_LLM_MODEL.get(model_name, {}).copy())
    # 更新config字典，将FSCHAT_MODEL_WORKERS中model_name对应的值合并到config中

    # 如果model_name在ONLINE_LLM_MODEL中
    if model_name in ONLINE_LLM_MODEL:
        # 设置config字典中的"online_api"为True
        config["online_api"] = True
        # 如果provider存在
        if provider := config.get("provider"):
            try:
                # 尝试获取model_workers模块中provider对应的类，并设置为config字典中的"worker_class"
                config["worker_class"] = getattr(model_workers, provider)
            except Exception as e:
                # 如果出现异常，记录错误信息
                msg = f"在线模型 ‘{model_name}’ 的provider没有正确配置"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
    
    # 如果model_name在MODEL_PATH["llm_model"]中
    if model_name in MODEL_PATH["llm_model"]:
        # 获取model_name对应的模型路径
        path = get_model_path(model_name)
        # 设置config字典中的"model_path"为path
        config["model_path"] = path
        # 如果path存在且是一个目录
        if path and os.path.isdir(path):
            # 设置config字典中的"model_path_exists"为True
            config["model_path_exists"] = True
        # 设置config字典中的"device"为llm_device函数处理后的结果
        config["device"] = llm_device(config.get("device"))
    
    # 返回config字典
    return config
# 获取所有模型工作配置信息的函数
def get_all_model_worker_configs() -> dict:
    result = {}
    # 获取所有模型名称
    model_names = set(FSCHAT_MODEL_WORKERS.keys())
    # 遍历每个模型名称
    for name in model_names:
        # 排除默认模型，获取该模型的工作配置信息并存入结果字典中
        if name != "default":
            result[name] = get_model_worker_config(name)
    return result


# 获取 fschat 控制器地址的函数
def fschat_controller_address() -> str:
    from configs.server_config import FSCHAT_CONTROLLER
    # 获取控制器的主机地址
    host = FSCHAT_CONTROLLER["host"]
    # 如果主机地址为 "0.0.0.0"，则替换为 "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    # 获取控制器的端口号
    port = FSCHAT_CONTROLLER["port"]
    return f"http://{host}:{port}"


# 获取 fschat 模型工作地址的函数
def fschat_model_worker_address(model_name: str = LLM_MODELS[0]) -> str:
    # 获取指定模型的工作配置信息
    if model := get_model_worker_config(model_name):
        # 获取模型工作地址的主机地址
        host = model["host"]
        # 如果主机地址为 "0.0.0.0"，则替换为 "127.0.0.1"
        if host == "0.0.0.0":
            host = "127.0.0.1"
        # 获取模型工作地址的端口号
        port = model["port"]
        return f"http://{host}:{port}"
    return ""


# 获取 fschat OpenAI API 地址的函数
def fschat_openai_api_address() -> str:
    from configs.server_config import FSCHAT_OPENAI_API
    # 获取 OpenAI API 的主机地址
    host = FSCHAT_OPENAI_API["host"]
    # 如果主机地址为 "0.0.0.0"，则替换为 "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    # 获取 OpenAI API 的端口号
    port = FSCHAT_OPENAI_API["port"]
    return f"http://{host}:{port}/v1"


# 获取 API 服务器地址的函数
def api_address() -> str:
    from configs.server_config import API_SERVER
    # 获取 API 服务器的主机地址
    host = API_SERVER["host"]
    # 如果主机地址为 "0.0.0.0"，则替换为 "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    # 获取 API 服务器的端口号
    port = API_SERVER["port"]
    return f"http://{host}:{port}"


# 获取 WebUI 服务器地址的函数
def webui_address() -> str:
    from configs.server_config import WEBUI_SERVER
    # 获取 WebUI 服务器的主机地址
    host = WEBUI_SERVER["host"]
    # 获取 WebUI 服务器的端口号
    port = WEBUI_SERVER["port"]
    return f"http://{host}:{port}"


# 获取提示模板的函数
def get_prompt_template(type: str, name: str) -> Optional[str]:
    '''
    从prompt_config中加载模板内容
    type: "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat"的其中一种，如果有新功能，应该进行加入。
    '''
    from configs import prompt_config
    import importlib
    importlib.reload(prompt_config)
    return prompt_config.PROMPT_TEMPLATES[type].get(name)


# 设置 HTTPX 配置的函数
def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
):
    '''
    # 设置httpx默认timeout，增加连接、读取、写入超时时间
    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 在进程范围内设置系统级代理
    proxies = {}
    # 如果proxy是字符串，则将其设置为http、https和all的代理
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    # 如果proxy是字典，则根据键值对设置代理
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    # 将代理信息设置到环境变量中
    for k, v in proxies.items():
        os.environ[k] = v

    # 设置不使用代理的主机列表
    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        # 不使用代理的本地主机
        "http://127.0.0.1",
        "http://localhost",
    ]
    # 不使用代理的用户部署的fastchat服务器
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        if host not in no_proxy:
            no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    # 定义一个函数用于获取代理信息
    def _get_proxies():
        return proxies

    # 重写urllib.request.getproxies函数，返回设置的代理信息
    import urllib.request
    urllib.request.getproxies = _get_proxies
# 检测设备类型，返回值为 "cuda", "mps", "cpu" 中的一个
def detect_device() -> Literal["cuda", "mps", "cpu"]:
    try:
        # 导入 torch 库
        import torch
        # 如果 CUDA 可用，则返回 "cuda"
        if torch.cuda.is_available():
            return "cuda"
        # 如果 MPS 可用，则返回 "mps"
        if torch.backends.mps.is_available():
            return "mps"
    except:
        pass
    # 默认返回 "cpu"
    return "cpu"


# 获取 LLModel 的设备类型，默认为 LLM_DEVICE，如果不在 ["cuda", "mps", "cpu"] 中，则调用 detect_device() 获取
def llm_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    device = device or LLM_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device


# 获取嵌入模型的设备类型，默认为 EMBEDDING_DEVICE，如果不在 ["cuda", "mps", "cpu"] 中，则调用 detect_device() 获取
def embedding_device(device: str = None) -> Literal["cuda", "mps", "cpu"]:
    device = device or EMBEDDING_DEVICE
    if device not in ["cuda", "mps", "cpu"]:
        device = detect_device()
    return device


# 在线程池中批量运行任务，并将运行结果以生成器的形式返回
def run_in_thread_pool(
        func: Callable,
        params: List[Dict] = [],
) -> Generator:
    '''
    在线程池中批量运行任务，并将运行结果以生成器的形式返回。
    请确保任务中的所有操作是线程安全的，任务函数请全部使用关键字参数。
    '''
    tasks = []
    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor() as pool:
        # 遍历参数列表，提交任务到线程池
        for kwargs in params:
            thread = pool.submit(func, **kwargs)
            tasks.append(thread)

        # 等待所有任务完成，返回结果
        for obj in as_completed(tasks):
            yield obj.result()


# 获取 httpx 客户端，支持异步请求，设置代理和超时时间
def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    '''
    helper to get httpx client with default proxies that bypass local addesses.
    '''
    default_proxies = {
        # 不使用代理访问本地地址
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # 不使用代理访问用户部署的 fastchat 服务器地址
    for x in [
        fschat_controller_address(),
        fschat_model_worker_address(),
        fschat_openai_api_address(),
    ]:
        host = ":".join(x.split(":")[:2])
        default_proxies.update({host: None})

    # 从系统环境中获取代理设置
    # 代理不是空字符串、None、False、0、空列表或空字典
    # 更新默认代理字典，根据环境变量设置的代理信息
    default_proxies.update({
        "http://": (os.environ.get("http_proxy")
                    if os.environ.get("http_proxy") and len(os.environ.get("http_proxy").strip())
                    else None),
        "https://": (os.environ.get("https_proxy")
                     if os.environ.get("https_proxy") and len(os.environ.get("https_proxy").strip())
                     else None),
        "all://": (os.environ.get("all_proxy")
                   if os.environ.get("all_proxy") and len(os.environ.get("all_proxy").strip())
                   else None),
    })
    # 遍历环境变量中的不需要代理的主机列表，更新默认代理字典
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # 原始代码
            default_proxies.update({'all://' + host: None})  # 修复 PR 1838，如果不添加 'all://'，httpx 将会报错

    # 将用户提供的代理信息与默认代理合并
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # 构建 Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    # 如果设置了日志详细信息，记录日志
    if log_verbose:
        logger.info(f'{get_httpx_client.__class__.__name__}:kwargs: {kwargs}')

    # 如果使用异步模式，返回异步 Client；否则返回同步 Client
    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)
# 获取configs中的原始配置项，供前端使用
def get_server_configs() -> Dict:
    # 导入需要的配置项
    from configs.kb_config import (
        DEFAULT_KNOWLEDGE_BASE,
        DEFAULT_SEARCH_ENGINE,
        DEFAULT_VS_TYPE,
        CHUNK_SIZE,
        OVERLAP_SIZE,
        SCORE_THRESHOLD,
        VECTOR_SEARCH_TOP_K,
        SEARCH_ENGINE_TOP_K,
        ZH_TITLE_ENHANCE,
        text_splitter_dict,
        TEXT_SPLITTER_NAME,
    )
    from configs.model_config import (
        LLM_MODELS,
        HISTORY_LEN,
        TEMPERATURE,
    )
    from configs.prompt_config import PROMPT_TEMPLATES

    # 自定义配置项
    _custom = {
        "controller_address": fschat_controller_address(),
        "openai_api_address": fschat_openai_api_address(),
        "api_address": api_address(),
    }

    # 返回所有配置项的字典
    return {**{k: v for k, v in locals().items() if k[0] != "_"}, **_custom}


# 列出在线嵌入模型
def list_online_embed_models() -> List[str]:
    # 导入模型工作类
    from server import model_workers

    ret = []
    # 遍历在线模型配置
    for k, v in list_config_llm_models()["online"].items():
        if provider := v.get("provider"):
            # 获取工作类
            worker_class = getattr(model_workers, provider, None)
            # 如果工作类存在且支持嵌入，则添加到返回列表中
            if worker_class is not None and worker_class.can_embedding():
                ret.append(k)
    return ret


# 加载本地嵌入
def load_local_embeddings(model: str = None, device: str = embedding_device()):
    # 导入嵌入池和嵌入模型配置
    from server.knowledge_base.kb_cache.base import embeddings_pool
    from configs import EMBEDDING_MODEL

    # 如果未指定模型，则使用默认嵌入模型
    model = model or EMBEDDING_MODEL
    return embeddings_pool.load_embeddings(model=model, device=device)


# 获取临时目录
def get_temp_dir(id: str = None) -> Tuple[str, str]:
    # 导入基本配置和临时文件模块
    from configs.basic_config import BASE_TEMP_DIR
    import tempfile

    # 如果指定的临时目录已存在，则直接返回
    if id is not None:
        path = os.path.join(BASE_TEMP_DIR, id)
        if os.path.isdir(path):
            return path, id

    # 创建临时目录并返回路径和文件夹名称
    path = tempfile.mkdtemp(dir=BASE_TEMP_DIR)
    return path, os.path.basename(path)
```