# `.\Langchain-Chatchat\startup.py`

```
# 导入必要的库
import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Process
from datetime import datetime
from pprint import pprint
from langchain_core._api import deprecated

# 尝试导入 numexpr 库，设置 NUMEXPR_MAX_THREADS 环境变量为检测到的 CPU 核心数
try:
    import numexpr
    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

# 将上级目录添加到 sys.path 中
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 从 configs 模块中导入所需的配置
from configs import (
    LOG_PATH,
    log_verbose,
    logger,
    LLM_MODELS,
    EMBEDDING_MODEL,
    TEXT_SPLITTER_NAME,
    FSCHAT_CONTROLLER,
    FSCHAT_OPENAI_API,
    FSCHAT_MODEL_WORKERS,
    API_SERVER,
    WEBUI_SERVER,
    HTTPX_DEFAULT_TIMEOUT,
)

# 从 server.utils 模块中导入所需的函数和类
from server.utils import (fschat_controller_address, fschat_model_worker_address,
                          fschat_openai_api_address, get_httpx_client, get_model_worker_config,
                          MakeFastAPIOffline, FastAPI, llm_device, embedding_device)

# 从 server.knowledge_base.migrate 模块中导入 create_tables 函数
from server.knowledge_base.migrate import create_tables

# 导入 argparse 模块和 List, Dict 类型
import argparse
from typing import List, Dict
from configs import VERSION

# 标记函数为废弃，提供相关信息
@deprecated(
    since="0.3.0",
    message="模型启动功能将于 Langchain-Chatchat 0.3.x重写,支持更多模式和加速启动，0.2.x中相关功能将废弃",
    removal="0.3.0")
def create_controller_app(
        dispatch_method: str,
        log_level: str = "INFO",
) -> FastAPI:
    # 设置日志目录
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    # 从 fastchat.serve.controller 模块中导入 app, Controller, logger
    from fastchat.serve.controller import app, Controller, logger
    # 设置日志级别
    logger.setLevel(log_level)

    # 创建 Controller 对象
    controller = Controller(dispatch_method)
    # 将 Controller 对象添加到模块中
    sys.modules["fastchat.serve.controller"].controller = controller

    # 将 FastAPI 应用设置为离线模式
    MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    app._controller = controller
    return app

# 创建模型工作进程的 FastAPI 应用
def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    kwargs包含的字段如下：
    host:
    port:
    model_names:[`model_name`]
    controller_address:
    worker_address:

    对于Langchain支持的模型：
        langchain_model:True
        不会使用fschat
    """
    # 设置在线API为True，并指定worker_class为provider
    对于online_api:
        online_api:True
        worker_class: `provider`
    
    # 对于离线模型，设置model_path为model_name_or_path，device为LLM_DEVICE
    对于离线模型：
        model_path: `model_name_or_path`,huggingface的repo-id或本地路径
        device:`LLM_DEVICE`
    
    """
    # 导入fastchat.constants模块，并设置LOGDIR为LOG_PATH
    import fastchat.constants
    fastchat.constants.LOGDIR = LOG_PATH
    # 导入argparse模块，创建ArgumentParser对象
    import argparse
    parser = argparse.ArgumentParser()
    # 解析空参数列表，返回args对象
    args = parser.parse_args([])

    # 遍历kwargs字典，将键值对设置为args对象的属性
    for k, v in kwargs.items():
        setattr(args, k, v)
    
    # 如果kwargs中存在"langchain_model"，则worker_class为该值，不做操作
    if worker_class := kwargs.get("langchain_model"):  
        from fastchat.serve.base_model_worker import app
        worker = ""
    
    # 如果kwargs中存在"worker_class"，则worker_class为该值，创建worker对象
    elif worker_class := kwargs.get("worker_class"):
        from fastchat.serve.base_model_worker import app
        worker = worker_class(model_names=args.model_names,
                              controller_addr=args.controller_address,
                              worker_addr=args.worker_address)
        # 设置日志级别为log_level
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    
    # 将app转为离线模型API
    MakeFastAPIOffline(app)
    # 设置app的标题为"FastChat LLM Server (第一个model_names)"
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    # 设置app的_worker属性为worker对象
    app._worker = worker
    # 返回app对象
    return app
# 创建并配置 OpenAI API 应用
def create_openai_api_app(
        controller_address: str,  # 控制器地址
        api_keys: List = [],  # API 密钥列表，默认为空列表
        log_level: str = "INFO",  # 日志级别，默认为 INFO
) -> FastAPI:  # 返回 FastAPI 实例
    import fastchat.constants  # 导入常量模块
    fastchat.constants.LOGDIR = LOG_PATH  # 设置日志目录
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings  # 导入相关模块
    from fastchat.utils import build_logger  # 导入构建日志器函数
    logger = build_logger("openai_api", "openai_api.log")  # 构建日志器
    logger.setLevel(log_level)  # 设置日志级别

    app.add_middleware(  # 添加中间件
        CORSMiddleware,  # 跨域中间件
        allow_credentials=True,  # 允许凭据
        allow_origins=["*"],  # 允许所有来源
        allow_methods=["*"],  # 允许所有方法
        allow_headers=["*"],  # 允许所有头部
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger  # 设置日志器
    app_settings.controller_address = controller_address  # 设置控制器地址
    app_settings.api_keys = api_keys  # 设置 API 密钥列表

    MakeFastAPIOffline(app)  # 将应用设置为离线状态
    app.title = "FastChat OpeanAI API Server"  # 设置应用标题
    return app  # 返回应用实例


# 设置应用事件
def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @app.on_event("startup")  # 在启动时执行
    async def on_startup():  # 启动时执行的异步函数
        if started_event is not None:  # 如果有启动事件
            started_event.set()  # 设置启动事件


# 运行控制器
def run_controller(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn  # 导入 uvicorn 模块
    import httpx  # 导入 httpx 模块
    from fastapi import Body  # 从 fastapi 模块导入 Body
    import time  # 导入 time 模块
    import sys  # 导入 sys 模块
    from server.utils import set_httpx_config  # 从 server.utils 导入设置 httpx 配置函数
    set_httpx_config()  # 设置 httpx 配置

    app = create_controller_app(  # 创建控制器应用
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),  # 调度方法
        log_level=log_level,  # 日志级别
    )
    _set_app_event(app, started_event)  # 设置应用事件

    # 添加释放和加载模型工作器的接口
    @app.post("/release_worker")  # POST 请求释放工作器
    def release_worker(
            model_name: str = Body(..., description="要释放模型的名称", samples=["chatglm-6b"]),  # 模型名称
            new_model_name: str = Body(None, description="释放后加载该模型"),  # 释放后加载的新模型名称
            keep_origin: bool = Body(False, description="不释放原模型，加载新模型")  # 是否保留原模型
    host = FSCHAT_CONTROLLER["host"]  # 主机地址
    port = FSCHAT_CONTROLLER["port"]  # 端口号
    # 如果日志级别为"ERROR"，则将标准输出和标准错误重定向回默认值
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # 运行 uvicorn 服务器，指定应用、主机、端口和日志级别（转换为小写）
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
# 定义一个运行模型工作进程的函数，接受模型名称、控制器地址、日志级别、队列和事件作为参数
def run_model_worker(
        model_name: str = LLM_MODELS[0],  # 默认模型名称为LLM_MODELS列表的第一个元素
        controller_address: str = "",  # 控制器地址默认为空字符串
        log_level: str = "INFO",  # 默认日志级别为INFO
        q: mp.Queue = None,  # 队列默认为空
        started_event: mp.Event = None,  # 事件默认为空
):
    import uvicorn  # 导入uvicorn模块
    from fastapi import Body  # 从fastapi模块导入Body类
    import sys  # 导入sys模块
    from server.utils import set_httpx_config  # 从server.utils模块导入set_httpx_config函数
    set_httpx_config()  # 调用set_httpx_config函数

    # 获取模型工作进程的配置信息
    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")  # 弹出配置信息中的host字段
    port = kwargs.pop("port")  # 弹出配置信息中的port字段
    kwargs["model_names"] = [model_name]  # 将模型名称添加到model_names字段中
    kwargs["controller_address"] = controller_address or fschat_controller_address()  # 设置控制器地址
    kwargs["worker_address"] = fschat_model_worker_address(model_name)  # 设置工作进程地址
    model_path = kwargs.get("model_path", "")  # 获取模型路径
    kwargs["model_path"] = model_path  # 设置模型路径

    # 创建模型工作进程的应用
    app = create_model_worker_app(log_level=log_level, **kwargs)
    _set_app_event(app, started_event)  # 设置应用事件
    if log_level == "ERROR":  # 如果日志级别为ERROR
        sys.stdout = sys.__stdout__  # 将标准输出重定向回原始标准输出
        sys.stderr = sys.__stderr__  # 将标准错误输出重定向回原始标准错误输出

    # 添加释放和加载模型的接口
    @app.post("/release")
    def release_model(
            new_model_name: str = Body(None, description="释放后加载该模型"),  # 新模型名称，默认为None
            keep_origin: bool = Body(False, description="不释放原模型，加载新模型")  # 是否保留原模型，默认为False
    ) -> Dict:  # 返回字典类型
        if keep_origin:  # 如果保留原模型
            if new_model_name:  # 如果有新模型名称
                q.put([model_name, "start", new_model_name])  # 将操作信息放入队列
        else:  # 如果不保留原模型
            if new_model_name:  # 如果有新模型名称
                q.put([model_name, "replace", new_model_name])  # 将操作信息放入队列
            else:  # 如果没有新模型名称
                q.put([model_name, "stop", None])  # 将停止操作信息放入队列
        return {"code": 200, "msg": "done"}  # 返回操作完成的消息

    # 运行应用
    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


# 定义一个运行OpenAI API的函数，接受日志级别和事件作为参数
def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):  # 默认日志级别为INFO，事件默认为空
    import uvicorn  # 导入uvicorn模块
    import sys  # 导入sys模块
    from server.utils import set_httpx_config  # 从server.utils模块导入set_httpx_config函数
    set_httpx_config()  # 调用set_httpx_config函数

    controller_addr = fschat_controller_address()  # 获取控制器地址
    app = create_openai_api_app(controller_addr, log_level=log_level)  # 创建OpenAI API应用
    _set_app_event(app, started_event)  # 设置应用事件

    host = FSCHAT_OPENAI_API["host"]  # 获取OpenAI API的主机地址
    # 从 FSCHAT_OPENAI_API 字典中获取端口号
    port = FSCHAT_OPENAI_API["port"]
    # 如果日志级别为"ERROR"，则将标准输出和标准错误重定向回默认值
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    # 运行 uvicorn 服务器，指定主机和端口
    uvicorn.run(app, host=host, port=port)
# 运行 API 服务器
def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    # 导入必要的模块和函数
    from server.api import create_app
    import uvicorn
    from server.utils import set_httpx_config
    # 设置 HTTPX 配置
    set_httpx_config()

    # 创建 API 应用
    app = create_app(run_mode=run_mode)
    # 设置应用事件
    _set_app_event(app, started_event)

    # 获取 API 服务器的主机和端口
    host = API_SERVER["host"]
    port = API_SERVER["port"]

    # 运行 API 服务器
    uvicorn.run(app, host=host, port=port)


# 运行 WebUI 服务器
def run_webui(started_event: mp.Event = None, run_mode: str = None):
    # 导入必要的模块和函数
    from server.utils import set_httpx_config
    # 设置 HTTPX 配置
    set_httpx_config()

    # 获取 WebUI 服务器的主机和端口
    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]

    # 构建运行 WebUI 的命令
    cmd = ["streamlit", "run", "webui.py",
           "--server.address", host,
           "--server.port", str(port),
           "--theme.base", "light",
           "--theme.primaryColor", "#165dff",
           "--theme.secondaryBackgroundColor", "#f5f5f5",
           "--theme.textColor", "#000000",
           ]
    # 如果运行模式为 "lite"，则添加额外参数
    if run_mode == "lite":
        cmd += [
            "--",
            "lite",
        ]
    # 启动子进程运行命令
    p = subprocess.Popen(cmd)
    # 设置事件为已启动
    started_event.set()
    # 等待子进程结束
    p.wait()


# 解析命令行参数
def parse_args() -> argparse.ArgumentParser:
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    # 添加参数选项
    parser.add_argument(
        "-a",
        "--all-webui",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py and webui.py",
        dest="all_webui",
    )
    parser.add_argument(
        "--all-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py",
        dest="all_api",
    )
    parser.add_argument(
        "--llm-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers",
        dest="llm_api",
    )
    parser.add_argument(
        "-o",
        "--openai-api",
        action="store_true",
        help="run fastchat's controller/openai_api servers",
        dest="openai_api",
    )
    # 添加参数 -m 或 --model-worker，设置为 True 表示运行 model_worker 服务器，并指定模型名称
    # 如果不使用默认的 LLM_MODELS，则需要指定 --model-name
    parser.add_argument(
        "-m",
        "--model-worker",
        action="store_true",
        help="run fastchat's model_worker server with specified model name. "
             "specify --model-name if not using default LLM_MODELS",
        dest="model_worker",
    )
    # 添加参数 -n 或 --model-name，指定模型名称给 model_worker
    # 默认为 LLM_MODELS，可以通过空格分隔添加多个模型名称
    parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        nargs="+",
        default=LLM_MODELS,
        help="specify model name for model worker. "
             "add addition names with space seperated to start multiple model workers.",
        dest="model_name",
    )
    # 添加参数 -c 或 --controller，指定 worker 注册的 controller 地址
    # 默认为 FSCHAT_CONTROLLER
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        help="specify controller address the worker is registered to. default is FSCHAT_CONTROLLER",
        dest="controller_address",
    )
    # 添加参数 --api，设置为 True 表示运行 api.py 服务器
    parser.add_argument(
        "--api",
        action="store_true",
        help="run api.py server",
        dest="api",
    )
    # 添加参数 -p 或 --api-worker，设置为 True 表示运行在线模型 API，如 zhipuai
    parser.add_argument(
        "-p",
        "--api-worker",
        action="store_true",
        help="run online model api such as zhipuai",
        dest="api_worker",
    )
    # 添加参数 -w 或 --webui，设置为 True 表示运行 webui.py 服务器
    parser.add_argument(
        "-w",
        "--webui",
        action="store_true",
        help="run webui.py server",
        dest="webui",
    )
    # 添加参数 -q 或 --quiet，设置为 True 减少 fastchat 服务的日志信息
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="减少fastchat服务log信息",
        dest="quiet",
    )
    # 添加参数 -i 或 --lite，设置为 True 以 Lite 模式运行
    # 仅支持在线 API 的 LLM 对话、搜索引擎对话
    parser.add_argument(
        "-i",
        "--lite",
        action="store_true",
        help="以Lite模式运行：仅支持在线API的LLM对话、搜索引擎对话",
        dest="lite",
    )
    # 解析命令行参数
    args = parser.parse_args()
    # 返回解析后的参数和参数解析器
    return args, parser
# 定义一个函数，用于打印服务器信息
def dump_server_info(after_start=False, args=None):
    # 导入必要的模块
    import platform
    import langchain
    import fastchat
    from server.utils import api_address, webui_address

    # 打印标题
    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    # 打印操作系统信息
    print(f"操作系统：{platform.platform()}.")
    # 打印Python版本信息
    print(f"python版本：{sys.version}")
    # 打印项目版本信息
    print(f"项目版本：{VERSION}")
    # 打印langchain和fastchat版本信息
    print(f"langchain版本：{langchain.__version__}. fastchat版本：{fastchat.__version__}")
    print("\n")

    # 设置LLM模型
    models = LLM_MODELS
    if args and args.model_name:
        models = args.model_name

    # 打印当前使用的分词器和LLM模型信息
    print(f"当前使用的分词器：{TEXT_SPLITTER_NAME}")
    print(f"当前启动的LLM模型：{models} @ {llm_device()}")

    # 遍历模型并打印模型的工作配置
    for model in models:
        pprint(get_model_worker_config(model))
    # 打印当前的Embbedings模型信息
    print(f"当前Embbedings模型： {EMBEDDING_MODEL} @ {embedding_device()}")

    # 如果是在启动后执行，打印服务端运行信息
    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.openai_api:
            print(f"    OpenAI API Server: {fschat_openai_api_address()}")
        if args.api:
            print(f"    Chatchat  API  Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    # 打印结束标志
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")

# 异步启动主服务器
async def start_main_server():
    # 导入必要的模块
    import time
    import signal

    # 信号处理函数
    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # 设置信号处理函数
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    # 设置进程启动方式
    mp.set_start_method("spawn")
    # 创建进程管理器
    manager = mp.Manager()
    run_mode = None

    # 创建队列
    queue = manager.Queue()
    # 解析命令行参数
    args, parser = parse_args()
    # 如果参数中包含 all_webui，则设置相关参数为 True
    if args.all_webui:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = True

    # 如果参数中包含 all_api，则设置相关参数为 True
    elif args.all_api:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = False

    # 如果参数中包含 llm_api，则设置相关参数为 True
    elif args.llm_api:
        args.openai_api = True
        args.model_worker = True
        args.api_worker = True
        args.api = False
        args.webui = False

    # 如果参数中包含 lite，则设置 model_worker 为 False，run_mode 为 "lite"
    if args.lite:
        args.model_worker = False
        run_mode = "lite"

    # 输出服务器信息
    dump_server_info(args=args)

    # 如果命令行参数数量大于 1，则输出启动服务信息
    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {LOG_PATH}")

    # 初始化进程字典
    processes = {"online_api": {}, "model_worker": {}}

    # 定义函数 process_count，返回进程数量
    def process_count():
        return len(processes) + len(processes["online_api"]) + len(processes["model_worker"]) - 2

    # 根据参数设置日志级别
    if args.quiet or not log_verbose:
        log_level = "ERROR"
    else:
        log_level = "INFO"

    # 创建 manager.Event 对象 controller_started
    controller_started = manager.Event()

    # 如果参数中包含 openai_api，则创建 controller 进程和 openai_api 进程
    if args.openai_api:
        process = Process(
            target=run_controller,
            name=f"controller",
            kwargs=dict(log_level=log_level, started_event=controller_started),
            daemon=True,
        )
        processes["controller"] = process

        process = Process(
            target=run_openai_api,
            name=f"openai_api",
            daemon=True,
        )
        processes["openai_api"] = process

    # 初始化 model_worker_started 列表
    model_worker_started = []
    # 如果设置了 model_worker 参数
    if args.model_worker:
        # 遍历每个模型名称
        for model_name in args.model_name:
            # 获取模型工作进程的配置信息
            config = get_model_worker_config(model_name)
            # 如果配置中没有在线 API
            if not config.get("online_api"):
                # 创建一个事件对象
                e = manager.Event()
                # 将事件对象添加到 model_worker_started 列表中
                model_worker_started.append(e)
                # 创建一个进程对象，用于运行模型工作进程
                process = Process(
                    target=run_model_worker,
                    name=f"model_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                # 将进程对象添加到 processes 字典中
                processes["model_worker"][model_name] = process

    # 如果设置了 api_worker 参数
    if args.api_worker:
        # 遍历每个模型名称
        for model_name in args.model_name:
            # 获取模型工作进程的配置信息
            config = get_model_worker_config(model_name)
            # 如果配置中有在线 API、工作类和模型名称在 FSCHAT_MODEL_WORKERS 中
            if (config.get("online_api")
                    and config.get("worker_class")
                    and model_name in FSCHAT_MODEL_WORKERS):
                # 创建一个事件对象
                e = manager.Event()
                # 将事件对象添加到 model_worker_started 列表中
                model_worker_started.append(e)
                # 创建一个进程对象，用于运行 API 工作进程
                process = Process(
                    target=run_model_worker,
                    name=f"api_worker - {model_name}",
                    kwargs=dict(model_name=model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e),
                    daemon=True,
                )
                # 将进程对象添加到 processes 字典中
                processes["online_api"][model_name] = process

    # 创建一个事件对象，用于标记 API 是否已启动
    api_started = manager.Event()
    # 如果设置了 api 参数
    if args.api:
        # 创建一个进程对象，用于运行 API 服务器
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=True,
        )
        # 将进程对象添加到 processes 字典中
        processes["api"] = process

    # 创建一个事件对象，用于标记 Web UI 是否已启动
    webui_started = manager.Event()
    # 如果参数中包含 webui，则创建一个进程对象，用于运行 webui 服务器
    if args.webui:
        # 创建一个进程对象，目标函数为 run_webui，进程名为 WEBUI Server
        # 传入参数为 started_event、run_mode，并设置为守护进程
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(started_event=webui_started, run_mode=run_mode),
            daemon=True,
        )
        # 将创建的进程对象存入进程字典中，键为 webui
        processes["webui"] = process

    # 如果当前没有任何进程在运行，则打印命令行参数帮助信息
    if process_count() == 0:
        parser.print_help()
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 创建数据库表
    create_tables()
    # 检查 Python 版本是否小于 3.10
    if sys.version_info < (3, 10):
        # 获取事件循环
        loop = asyncio.get_event_loop()
    else:
        try:
            # 尝试获取正在运行的事件循环
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 如果没有正在运行的事件循环，则创建一个新的事件循环
            loop = asyncio.new_event_loop()

        # 设置事件循环
        asyncio.set_event_loop(loop)

    # 运行主服务器
    loop.run_until_complete(start_main_server())

# 服务启动后接口调用示例：
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm3-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
```