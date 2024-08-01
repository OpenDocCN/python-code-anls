# `.\DB-GPT-src\dbgpt\app\dbgpt_server.py`

```py
import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统相关功能模块
import sys  # 导入系统相关功能模块
from typing import List  # 引入类型提示模块中的List类型

from fastapi import FastAPI  # 导入FastAPI框架
from fastapi.middleware.cors import CORSMiddleware  # 导入FastAPI的CORS中间件

# fastapi import time cost about 0.05s
from fastapi.staticfiles import StaticFiles  # 导入FastAPI静态文件处理模块

from dbgpt._private.config import Config  # 导入私有配置模块Config
from dbgpt._version import version  # 导入版本信息
from dbgpt.app.base import (  # 导入应用基础模块
    WebServerParameters,  # Web服务器参数
    _create_model_start_listener,  # 创建模型启动监听器
    _migration_db_storage,  # 数据库存储迁移
    server_init,  # 服务器初始化
)

# initialize_components import time cost about 0.1s
from dbgpt.app.component_configs import initialize_components  # 导入组件配置初始化模块
from dbgpt.component import SystemApp  # 导入SystemApp组件
from dbgpt.configs.model_config import (  # 导入模型配置
    EMBEDDING_MODEL_CONFIG,  # 嵌入模型配置
    LLM_MODEL_CONFIG,  # LLM模型配置
    LOGDIR,  # 日志目录
    STATIC_MESSAGE_IMG_PATH,  # 静态消息图片路径
)
from dbgpt.serve.core import add_exception_handler  # 导入异常处理器
from dbgpt.util.fastapi import create_app, replace_router  # 导入FastAPI工具函数
from dbgpt.util.i18n_utils import _, set_default_language  # 导入国际化工具函数
from dbgpt.util.parameter_utils import _get_dict_from_obj  # 导入参数工具函数
from dbgpt.util.system_utils import get_system_info  # 导入系统信息工具函数
from dbgpt.util.tracer import (  # 导入追踪器工具函数
    SpanType,  # 跨度类型
    SpanTypeRunName,  # 跨度类型运行名称
    initialize_tracer,  # 初始化追踪器
    root_tracer,  # 根追踪器
)
from dbgpt.util.utils import (  # 导入通用工具函数
    _get_logging_level,  # 获取日志级别
    logging_str_to_uvicorn_level,  # 将日志字符串转换为uvicorn日志级别
    setup_http_service_logging,  # 设置HTTP服务日志记录
    setup_logging,  # 设置日志记录
)

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

static_file_path = os.path.join(ROOT_PATH, "dbgpt", "app/static")  # 静态文件路径

CFG = Config()  # 创建Config对象
set_default_language(CFG.LANGUAGE)  # 设置默认语言

app = create_app(  # 创建FastAPI应用
    title=_("DB-GPT Open API"),  # API标题
    description=_("DB-GPT Open API"),  # API描述
    version=version,  # 版本号
    openapi_tags=[],  # 开放API标签
)
# Use custom router to support priority
replace_router(app)  # 替换路由器

app.mount(
    "/swagger_static",  # 挂载路径
    StaticFiles(directory=static_file_path),  # 静态文件目录
    name="swagger_static",  # 名称
)

system_app = SystemApp(app)  # 创建SystemApp对象


def mount_routers(app: FastAPI):
    """Lazy import to avoid high time cost"""
    from dbgpt.app.knowledge.api import router as knowledge_router  # 导入知识API路由
    from dbgpt.app.llm_manage.api import router as llm_manage_api  # 导入LLM管理API路由
    from dbgpt.app.openapi.api_v1.api_v1 import router as api_v1  # 导入API v1路由
    from dbgpt.app.openapi.api_v1.editor.api_editor_v1 import (  # 导入API编辑器 v1路由
        router as api_editor_route_v1,
    )
    from dbgpt.app.openapi.api_v1.feedback.api_fb_v1 import router as api_fb_v1  # 导入API反馈 v1路由
    from dbgpt.app.openapi.api_v2 import router as api_v2  # 导入API v2路由
    from dbgpt.serve.agent.app.controller import router as gpts_v1  # 导入GPTs应用控制器路由
    from dbgpt.serve.agent.app.endpoints import router as app_v2  # 导入应用 v2路由

    app.include_router(api_v1, prefix="/api", tags=["Chat"])  # 注册API v1路由
    app.include_router(api_v2, prefix="/api", tags=["ChatV2"])  # 注册API v2路由
    app.include_router(api_editor_route_v1, prefix="/api", tags=["Editor"])  # 注册编辑器API v1路由
    app.include_router(llm_manage_api, prefix="/api", tags=["LLM Manage"])  # 注册LLM管理API路由
    app.include_router(api_fb_v1, prefix="/api", tags=["FeedBack"])  # 注册反馈API v1路由
    app.include_router(gpts_v1, prefix="/api", tags=["GptsApp"])  # 注册GPTs应用控制器路由
    app.include_router(app_v2, prefix="/api", tags=["App"])  # 注册应用 v2路由

    app.include_router(knowledge_router, tags=["Knowledge"])  # 注册知识API路由
def mount_static_files(app: FastAPI):
    # 确保静态消息图像路径存在，如果不存在则创建
    os.makedirs(STATIC_MESSAGE_IMG_PATH, exist_ok=True)
    # 将指定目录下的图像文件映射到"/images"路径下，启用HTML兼容
    app.mount(
        "/images",
        StaticFiles(directory=STATIC_MESSAGE_IMG_PATH, html=True),
        name="static2",
    )
    # 将指定目录下的"_next/static"文件映射到"/_next/static"路径下
    app.mount(
        "/_next/static", StaticFiles(directory=static_file_path + "/_next/static")
    )
    # 将指定目录下的所有文件映射到根路径"/"下，启用HTML兼容
    app.mount("/", StaticFiles(directory=static_file_path, html=True), name="static")


add_exception_handler(app)


def _get_webserver_params(args: List[str] = None):
    # 导入环境参数解析器
    from dbgpt.util.parameter_utils import EnvArgumentParser

    parser = EnvArgumentParser()

    env_prefix = "webserver_"
    # 从环境变量和命令行参数中解析Web服务器参数并返回数据类实例
    webserver_params: WebServerParameters = parser.parse_args_into_dataclass(
        WebServerParameters,
        env_prefixes=[env_prefix],
        command_args=args,
    )
    return webserver_params


def initialize_app(param: WebServerParameters = None, args: List[str] = None):
    """Initialize app
    If you use gunicorn as a process manager, initialize_app can be invoke in `on_starting` hook.
    Args:
        param:WebWerverParameters
        args:List[str]
    """
    # 如果参数为空，则从命令行获取Web服务器参数
    if not param:
        param = _get_webserver_params(args)

    # 在参数初始化后导入模块，加快--help的响应速度
    from dbgpt.model.cluster import initialize_worker_manager_in_client

    # 如果参数中没有指定日志级别，则获取默认日志级别
    if not param.log_level:
        param.log_level = _get_logging_level()
    # 设置日志记录器的名称、日志级别和日志文件名
    setup_logging(
        "dbgpt", logging_level=param.log_level, logger_filename=param.log_file
    )

    # 获取模型名称，若未指定则使用默认的LLM_MODEL配置
    model_name = param.model_name or CFG.LLM_MODEL
    param.model_name = model_name
    # 如果端口未指定，则使用默认的DBGPT_WEBSERVER_PORT端口号
    param.port = param.port or CFG.DBGPT_WEBSERVER_PORT
    # 若端口为空，则设置为5670端口
    if not param.port:
        param.port = 5670

    # 打印参数信息
    print(param)

    # 获取嵌入模型名称和路径
    embedding_model_name = CFG.EMBEDDING_MODEL
    embedding_model_path = EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
    rerank_model_name = CFG.RERANK_MODEL
    rerank_model_path = None
    # 如果存在重排序模型名称，则获取重排序模型路径
    if rerank_model_name:
        rerank_model_path = CFG.RERANK_MODEL_PATH or EMBEDDING_MODEL_CONFIG.get(
            rerank_model_name
        )

    # 初始化服务器，传入参数和应用实例
    server_init(param, system_app)
    # 挂载路由
    mount_routers(app)
    # 创建模型启动监听器，并返回监听器实例
    model_start_listener = _create_model_start_listener(system_app)
    # 初始化组件，包括参数、应用实例、嵌入模型和重排序模型等信息
    initialize_components(
        param,
        system_app,
        embedding_model_name,
        embedding_model_path,
        rerank_model_name,
        rerank_model_path,
    )
    # 执行系统应用的初始化操作
    system_app.on_init()

    # 数据库迁移，需要先导入数据库模型
    _migration_db_storage(param)

    # 获取模型路径，若未指定则使用LLM_MODEL_PATH配置或LLM_MODEL_CONFIG中对应模型路径
    model_path = CFG.LLM_MODEL_PATH or LLM_MODEL_CONFIG.get(model_name)
    # TODO: 将initialize_worker_manager_in_client作为组件注册到system_app
    # 如果不是轻量模式，则打印统一部署模式的信息
    if not param.light:
        print("Model Unified Deployment Mode!")
        
        # 如果不是远程嵌入模型，则将嵌入模型的名称和路径设置为None
        if not param.remote_embedding:
            embedding_model_name, embedding_model_path = None, None
        
        # 如果不是远程重排序模型，则将重排序模型的名称和路径设置为None
        if not param.remote_rerank:
            rerank_model_name, rerank_model_path = None, None
        
        # 在客户端初始化工作管理器
        initialize_worker_manager_in_client(
            app=app,
            model_name=model_name,
            model_path=model_path,
            local_port=param.port,
            embedding_model_name=embedding_model_name,
            embedding_model_path=embedding_model_path,
            rerank_model_name=rerank_model_name,
            rerank_model_path=rerank_model_path,
            start_listener=model_start_listener,
            system_app=system_app,
        )

        # 设置NEW_SERVER_MODE为True，表示新的服务器模式
        CFG.NEW_SERVER_MODE = True
    else:
        # 否则，模型服务器地址为控制器地址或默认地址CFG.MODEL_SERVER
        controller_addr = param.controller_addr or CFG.MODEL_SERVER
        
        # 在客户端初始化工作管理器，设置为运行在远程模式
        initialize_worker_manager_in_client(
            app=app,
            model_name=model_name,
            model_path=model_path,
            run_locally=False,
            controller_addr=controller_addr,
            local_port=param.port,
            start_listener=model_start_listener,
            system_app=system_app,
        )
        
        # 设置SERVER_LIGHT_MODE为True，表示服务器轻量模式
        CFG.SERVER_LIGHT_MODE = True

    # 挂载静态文件到应用中
    mount_static_files(app)

    # 在启动之前执行系统应用的before_start方法
    system_app.before_start()
    
    # 返回参数param
    return param
# 如果没有传入参数，使用默认的 WebServerParameters
if not param:
    param = _get_webserver_params()

# 初始化追踪器，配置追踪器的各项参数和选项
initialize_tracer(
    os.path.join(LOGDIR, param.tracer_file),  # 设置追踪器日志文件路径
    system_app=system_app,  # 设置系统应用
    tracer_storage_cls=param.tracer_storage_cls,  # 设置追踪器存储类
    enable_open_telemetry=param.tracer_to_open_telemetry,  # 是否启用开放遥测
    otlp_endpoint=param.otel_exporter_otlp_traces_endpoint,  # OTLP 追踪数据导出端点
    otlp_insecure=param.otel_exporter_otlp_traces_insecure,  # OTLP 连接是否不安全
    otlp_timeout=param.otel_exporter_otlp_traces_timeout,  # OTLP 连接超时时间
)

# 使用根追踪器开始一个名为 "run_webserver" 的新追踪 span
with root_tracer.start_span(
    "run_webserver",  # 设置 span 的名称为 "run_webserver"
    span_type=SpanType.RUN,  # 设置 span 的类型为运行类型
    metadata={
        "run_service": SpanTypeRunName.WEBSERVER,  # 设置 span 的服务类型为 WEBSERVER
        "params": _get_dict_from_obj(param),  # 将 param 对象转换为字典存入 metadata
        "sys_infos": _get_dict_from_obj(get_system_info()),  # 获取系统信息转换为字典存入 metadata
    },
):
    # 初始化应用程序
    param = initialize_app(param)
    # 运行 uvicorn 服务器，传入参数 param
    run_uvicorn(param)
```