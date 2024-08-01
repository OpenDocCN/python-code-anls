# `.\DB-GPT-src\dbgpt\model\cli.py`

```py
import functools  # 导入functools模块，用于高阶函数的操作
import logging  # 导入logging模块，用于日志记录
import os  # 导入os模块，提供了与操作系统交互的功能
from typing import Callable, List, Optional, Type  # 导入类型提示相关的功能

import click  # 导入click库，用于创建命令行接口

from dbgpt.configs.model_config import LOGDIR  # 从dbgpt.configs.model_config导入LOGDIR常量
from dbgpt.model.base import WorkerApplyType  # 导入WorkerApplyType类
from dbgpt.model.parameter import (  # 导入参数相关类
    BaseParameters,
    ModelAPIServerParameters,
    ModelControllerParameters,
    ModelParameters,
    ModelWorkerParameters,
)
from dbgpt.util import get_or_create_event_loop  # 导入获取或创建事件循环的函数
from dbgpt.util.command_utils import (  # 导入命令行工具相关的函数
    _detect_controller_address,
    _run_current_with_daemon,
    _stop_service,
)
from dbgpt.util.parameter_utils import (  # 导入参数工具相关的函数和类
    EnvArgumentParser,
    _build_parameter_class,
    build_lazy_click_command,
)

# 可通过设置环境变量CONTROLLER_ADDRESS来配置默认地址
MODEL_CONTROLLER_ADDRESS = "http://127.0.0.1:8000"

logger = logging.getLogger("dbgpt_cli")  # 获取名为"dbgpt_cli"的日志记录器


def _get_worker_manager(address: str):
    """获取工作管理器对象

    Args:
        address (str): 控制器地址

    Returns:
        RemoteWorkerManager: 远程工作管理器对象
    """
    from dbgpt.model.cluster import ModelRegistryClient, RemoteWorkerManager

    registry = ModelRegistryClient(address)  # 创建模型注册客户端对象
    worker_manager = RemoteWorkerManager(registry)  # 创建远程工作管理器对象
    return worker_manager


@click.group("model")
@click.option(
    "--address",
    type=str,
    default=None,
    required=False,
    show_default=True,
    help=(
        "Address of the Model Controller to connect to. "
        "Just support light deploy model, If the environment variable CONTROLLER_ADDRESS is configured, read from the environment variable"
    ),
)
def model_cli_group(address: str):
    """Clients that manage model serving
    
    Args:
        address (str): Optional; 模型控制器的地址
    """
    global MODEL_CONTROLLER_ADDRESS
    if not address:
        MODEL_CONTROLLER_ADDRESS = _detect_controller_address()  # 检测控制器地址
    else:
        MODEL_CONTROLLER_ADDRESS = address  # 设置模型控制器地址


@model_cli_group.command()
@click.option(
    "--model_name", type=str, default=None, required=False, help=("The name of model")
)
@click.option(
    "--model_type", type=str, default="llm", required=False, help=("The type of model")
)
def list(model_name: str, model_type: str):
    """List model instances
    
    Args:
        model_name (str): Optional; 模型名称
        model_type (str): Optional; 模型类型，默认为"llm"
    """
    from prettytable import PrettyTable  # 导入PrettyTable用于创建漂亮的表格

    from dbgpt.model.cluster import ModelRegistryClient  # 导入模型注册客户端类

    loop = get_or_create_event_loop()  # 获取或创建事件循环
    registry = ModelRegistryClient(MODEL_CONTROLLER_ADDRESS)  # 创建模型注册客户端对象

    if not model_name:
        instances = loop.run_until_complete(registry.get_all_model_instances())  # 获取所有模型实例
    else:
        if not model_type:
            model_type = "llm"
        register_model_name = f"{model_name}@{model_type}"
        instances = loop.run_until_complete(
            registry.get_all_instances(register_model_name)
        )  # 获取指定模型名称和类型的实例

    table = PrettyTable()  # 创建表格对象

    table.field_names = [  # 设置表格的列名
        "Model Name",
        "Model Type",
        "Host",
        "Port",
        "Healthy",
        "Enabled",
        "Prompt Template",
        "Last Heartbeat",
    ]
    # 遍历 instances 列表中的每一个实例对象
    for instance in instances:
        # 将 model_name 字符串按 "@" 分割，得到模型名称和类型
        model_name, model_type = instance.model_name.split("@")
        # 向表格 table 中添加一行数据，包括模型名称、类型、主机、端口、健康状态、启用状态、
        # 提示模板（如果存在）、上次心跳时间等信息
        table.add_row(
            [
                model_name,
                model_type,
                instance.host,
                instance.port,
                instance.healthy,
                instance.enabled,
                instance.prompt_template if instance.prompt_template else "",  # 如果存在提示模板则添加，否则添加空字符串
                instance.last_heartbeat,
            ]
        )

    # 打印完整的表格
    print(table)
def add_model_options(func):
    # 装饰器函数，为给定函数添加模型选项
    @click.option(
        "--model_name",
        type=str,
        default=None,
        required=True,
        help=("The name of model"),
    )
    @click.option(
        "--model_type",
        type=str,
        default="llm",
        required=False,
        help=("The type of model"),
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 执行被装饰的函数，并传递参数
        return func(*args, **kwargs)

    return wrapper


@model_cli_group.command()
@add_model_options
@click.option(
    "--host",
    type=str,
    required=True,
    help=("The remote host to stop model"),
)
@click.option(
    "--port",
    type=int,
    required=True,
    help=("The remote port to stop model"),
)
def stop(model_name: str, model_type: str, host: str, port: int):
    """Stop model instances"""
    from dbgpt.model.cluster import RemoteWorkerManager, WorkerStartupRequest

    # 获取远程工作管理器对象
    worker_manager: RemoteWorkerManager = _get_worker_manager(MODEL_CONTROLLER_ADDRESS)
    # 创建工作启动请求对象
    req = WorkerStartupRequest(
        host=host,
        port=port,
        worker_type=model_type,
        model=model_name,
        params={},
    )
    # 获取或创建事件循环
    loop = get_or_create_event_loop()
    # 运行直到完成模型关闭操作，获取结果
    res = loop.run_until_complete(worker_manager.model_shutdown(req))
    # 打印操作结果
    print(res)


def _remote_model_dynamic_factory() -> Callable[[None], List[Type]]:
    # 导入所需模块和类
    from dataclasses import dataclass, field
    from dbgpt.model.cluster import RemoteWorkerManager
    from dbgpt.model.parameter import WorkerType
    from dbgpt.util.parameter_utils import _SimpleArgParser

    # 使用简单参数解析器解析预设参数
    pre_args = _SimpleArgParser("model_name", "address", "host", "port")
    pre_args.parse()
    model_name = pre_args.get("model_name")
    address = pre_args.get("address")
    host = pre_args.get("host")
    port = pre_args.get("port")
    if port:
        port = int(port)

    # 如果未指定地址，自动探测控制器地址
    if not address:
        address = _detect_controller_address()

    # 获取远程工作管理器对象
    worker_manager: RemoteWorkerManager = _get_worker_manager(address)
    # 获取或创建事件循环
    loop = get_or_create_event_loop()
    # 运行直到完成获取支持的模型列表操作
    models = loop.run_until_complete(worker_manager.supported_models())

    # 定义字段字典，用于动态创建数据类
    fields_dict = {}
    fields_dict["model_name"] = (
        str,
        field(default=None, metadata={"help": "The model name to deploy"}),
    )
    fields_dict["host"] = (
        str,
        field(default=None, metadata={"help": "The remote host to deploy model"}),
    )
    fields_dict["port"] = (
        int,
        field(default=None, metadata={"help": "The remote port to deploy model"}),
    )
    # 动态创建数据类 RemoteModelWorkerParameters
    result_class = dataclass(
        type("RemoteModelWorkerParameters", (object,), fields_dict)
    )

    # 如果没有支持的模型，直接返回结果类列表
    if not models:
        return [result_class]

    # 初始化有效模型列表和相应的参数类列表
    valid_models = []
    valid_model_cls = []
    for model in models:
        # 过滤与指定主机和端口不匹配的模型
        if host and host != model.host:
            continue
        if port and port != model.port:
            continue
        # 添加有效模型到列表
        valid_models += [m.model for m in model.models]
        # 构建有效模型参数类列表
        valid_model_cls += [
            (m, _build_parameter_class(m.params)) for m in model.models if m.params
        ]
    # 从有效的模型类列表中获取第一个模型和其对应的参数类
    real_model, real_params_cls = valid_model_cls[0]
    # 初始化真实路径为None，工作类型为"llm"
    real_path = None
    real_worker_type = "llm"
    # 如果指定了模型名称
    if model_name:
        # 从有效的模型类列表中找到模型名称匹配的模型和参数类
        params_cls_list = [m for m in valid_model_cls if m[0].model == model_name]
        # 如果没有找到匹配的模型名称，则抛出数值错误异常
        if not params_cls_list:
            raise ValueError(f"Not supported model with model name: {model_name}")
        # 获取第一个匹配到的模型和参数类
        real_model, real_params_cls = params_cls_list[0]
        # 设置真实路径为匹配到的模型的路径
        real_path = real_model.path
        # 设置真实工作类型为匹配到的模型的工作类型

        real_worker_type = real_model.worker_type

    @dataclass
    # 定义一个数据类 RemoteModelWorkerParameters，继承自 BaseParameters
    class RemoteModelWorkerParameters(BaseParameters):
        model_name: str = field(
            metadata={"valid_values": valid_models, "help": "The model name to deploy"}
        )
        model_path: Optional[str] = field(
            default=real_path, metadata={"help": "The model path to deploy"}
        )
        host: Optional[str] = field(
            # 设置默认主机为第一个模型的主机地址
            default=models[0].host,
            metadata={
                "valid_values": [model.host for model in models],
                "help": "The remote host to deploy model",
            },
        )

        port: Optional[int] = field(
            # 设置默认端口为第一个模型的端口号
            default=models[0].port,
            metadata={
                "valid_values": [model.port for model in models],
                "help": "The remote port to deploy model",
            },
        )
        worker_type: Optional[str] = field(
            # 设置默认工作类型为真实的工作类型
            default=real_worker_type,
            metadata={
                "valid_values": WorkerType.values(),
                "help": "Worker type",
            },
        )

    # 返回 RemoteModelWorkerParameters 类和真实的参数类 real_params_cls 列表
    return [RemoteModelWorkerParameters, real_params_cls]
# 定义一个 CLI 命令，用于启动模型实例
@model_cli_group.command(
    cls=build_lazy_click_command(_dynamic_factory=_remote_model_dynamic_factory)
)
def start(**kwargs):
    """Start model instances"""
    # 导入必要的模块和类
    from dbgpt.model.cluster import RemoteWorkerManager, WorkerStartupRequest
    
    # 获取远程工作管理器实例
    worker_manager: RemoteWorkerManager = _get_worker_manager(MODEL_CONTROLLER_ADDRESS)
    
    # 创建 WorkerStartupRequest 请求对象，用于启动工作节点
    req = WorkerStartupRequest(
        host=kwargs["host"],
        port=kwargs["port"],
        worker_type=kwargs["worker_type"],
        model=kwargs["model_name"],
        params={},
    )
    
    # 删除不再需要的参数
    del kwargs["host"]
    del kwargs["port"]
    del kwargs["worker_type"]
    
    # 将剩余的 kwargs 参数设置为 req 的 params 属性
    req.params = kwargs
    
    # 获取或创建事件循环
    loop = get_or_create_event_loop()
    
    # 执行 worker_manager 的 model_startup 方法，并阻塞直到完成
    res = loop.run_until_complete(worker_manager.model_startup(req))
    
    # 打印启动模型实例的结果
    print(res)


# 定义一个 CLI 命令，用于重启模型实例
@model_cli_group.command()
@add_model_options
def restart(model_name: str, model_type: str):
    """Restart model instances"""
    # 调用 worker_apply 函数，重启指定的模型实例
    worker_apply(
        MODEL_CONTROLLER_ADDRESS, model_name, model_type, WorkerApplyType.RESTART
    )


# 定义一个 CLI 命令，用于与机器人进行命令行交互
@model_cli_group.command()
@click.option(
    "--model_name",
    type=str,
    default=None,
    required=True,
    help=("The name of model"),
)
@click.option(
    "--system",
    type=str,
    default=None,
    required=False,
    help=("System prompt"),
)
def chat(model_name: str, system: str):
    """Interact with your bot from the command line"""
    # 调用 _cli_chat 函数，与指定模型的机器人进行交互
    _cli_chat(MODEL_CONTROLLER_ADDRESS, model_name, system)


# 定义一个函数，用于向工作节点应用操作（如重启）
def worker_apply(
    address: str, model_name: str, model_type: str, apply_type: WorkerApplyType
):
    # 导入必要的模块和类
    from dbgpt.model.cluster import WorkerApplyRequest
    
    # 获取或创建事件循环
    loop = get_or_create_event_loop()
    
    # 获取工作管理器实例
    worker_manager = _get_worker_manager(address)
    
    # 创建 WorkerApplyRequest 请求对象，用于对工作节点应用操作
    apply_req = WorkerApplyRequest(
        model=model_name, worker_type=model_type, apply_type=apply_type
    )
    
    # 执行 worker_manager 的 worker_apply 方法，并阻塞直到完成
    res = loop.run_until_complete(worker_manager.worker_apply(apply_req))
    
    # 打印应用操作的结果
    print(res)


# 定义一个函数，用于在命令行中与指定模型的机器人进行交互
def _cli_chat(address: str, model_name: str, system_prompt: str = None):
    # 获取或创建事件循环
    loop = get_or_create_event_loop()
    
    # 获取工作管理器实例
    worker_manager = worker_manager = _get_worker_manager(address)
    
    # 执行 _chat_stream 函数，与指定模型的机器人进行交互
    loop.run_until_complete(_chat_stream(worker_manager, model_name, system_prompt))


# 异步函数，用于在命令行中与指定模型的机器人进行交互
async def _chat_stream(worker_manager, model_name: str, system_prompt: str = None):
    # 导入必要的模块和类
    from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType
    from dbgpt.model.cluster import PromptRequest
    
    # 输出信息，指示聊天机器人已启动
    print(f"Chatbot started with model {model_name}. Type 'exit' to leave the chat.")
    
    # 初始化历史记录和前一个响应
    hist = []
    previous_response = ""
    
    # 如果提供了系统提示，将其添加到历史记录中
    if system_prompt:
        hist.append(
            ModelMessage(role=ModelMessageRoleType.SYSTEM, content=system_prompt)
        )
    # 循环开始，持续接收用户输入并进行处理
    while True:
        # 初始化前一个响应为空字符串
        previous_response = ""
        # 获取用户输入
        user_input = input("\n\nYou: ")
        # 如果用户输入是"exit"，则退出循环
        if user_input.lower().strip() == "exit":
            break
        # 将用户输入添加到历史消息列表中，标记为人类角色
        hist.append(ModelMessage(role=ModelMessageRoleType.HUMAN, content=user_input))
        # 创建用于生成响应的请求对象，包括历史消息、模型名称等信息
        request = PromptRequest(messages=hist, model=model_name, prompt="", echo=False)
        # 将请求对象转换为字典形式（排除空值）
        request = request.dict(exclude_none=True)
        # 打印"Bot: "，准备输出机器人的响应
        print("Bot: ", end="")
        # 异步生成器，逐条获取响应消息流
        async for response in worker_manager.generate_stream(request):
            # 计算增量输出，即与前一个响应之后的新文本部分
            incremental_output = response.text[len(previous_response):]
            # 输出增量部分，不换行并刷新输出
            print(incremental_output, end="", flush=True)
            # 更新前一个响应文本为当前响应文本
            previous_response = response.text
        # 将最终机器人的响应添加到历史消息列表中，标记为AI角色
        hist.append(
            ModelMessage(role=ModelMessageRoleType.AI, content=previous_response)
        )
# 为给定的函数 `func` 添加停止服务器选项
def add_stop_server_options(func):
    # 使用 `click.option` 装饰器添加 `--port` 选项，指定类型为整数，默认为None，非必需，帮助信息为停止的端口号
    @click.option(
        "--port",
        type=int,
        default=None,
        required=False,
        help=("The port to stop"),
    )
    # 使用 `functools.wraps` 保留原始函数的元数据
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 调用原始函数 `func`，传递所有位置参数和关键字参数
        return func(*args, **kwargs)

    return wrapper


# 创建名为 `controller` 的点击命令，使用 `EnvArgumentParser.create_click_option` 设置参数
@click.command(name="controller")
@EnvArgumentParser.create_click_option(ModelControllerParameters)
def start_model_controller(**kwargs):
    """Start model controller"""

    # 如果 `daemon` 在参数中为真
    if kwargs["daemon"]:
        # 设置日志文件路径为 LOGDIR 下的 `model_controller_uvicorn.log`
        log_file = os.path.join(LOGDIR, "model_controller_uvicorn.log")
        # 使用 `_run_current_with_daemon` 启动名为 `ModelController` 的进程，并记录日志
        _run_current_with_daemon("ModelController", log_file)
    else:
        # 导入 `dbgpt.model.cluster` 下的 `run_model_controller` 函数并运行之
        from dbgpt.model.cluster import run_model_controller

        run_model_controller()


# 创建名为 `controller` 的点击命令，使用 `add_stop_server_options` 装饰器添加停止服务器选项
@click.command(name="controller")
@add_stop_server_options
def stop_model_controller(port: int):
    """Start model controller"""
    # 构建命令片段以检查正在运行的进程
    _stop_service("controller", "ModelController", port=port)


# 返回可调用对象的列表生成器函数 `_model_dynamic_factory`，返回类型为 `Callable[[None], List[Type]]`
def _model_dynamic_factory() -> Callable[[None], List[Type]]:
    # 导入 `dbgpt.model.adapter.model_adapter` 下的 `_dynamic_model_parser` 函数
    from dbgpt.model.adapter.model_adapter import _dynamic_model_parser

    # 调用 `_dynamic_model_parser` 函数并将结果赋给 `param_class`
    param_class = _dynamic_model_parser()
    # 设置 `fix_class` 为列表 `[ModelWorkerParameters]`
    fix_class = [ModelWorkerParameters]
    # 如果 `param_class` 为假值，则将 `[ModelParameters]` 添加到 `fix_class` 中
    if not param_class:
        param_class = [ModelParameters]
    fix_class += param_class
    # 返回 `fix_class`
    return fix_class


# 创建名为 `worker` 的点击命令，使用 `build_lazy_click_command` 构建懒加载点击命令，使用 `_model_dynamic_factory` 作为动态工厂
@click.command(
    name="worker", cls=build_lazy_click_command(_dynamic_factory=_model_dynamic_factory)
)
def start_model_worker(**kwargs):
    """Start model worker"""
    # 如果 `daemon` 在参数中为真
    if kwargs["daemon"]:
        # 设置端口号为 `port` 参数，模型类型为 `worker_type` 参数或默认为 "llm"
        port = kwargs["port"]
        model_type = kwargs.get("worker_type") or "llm"
        # 设置日志文件路径为 LOGDIR 下的 `model_worker_{model_type}_{port}_uvicorn.log`
        log_file = os.path.join(LOGDIR, f"model_worker_{model_type}_{port}_uvicorn.log")
        # 使用 `_run_current_with_daemon` 启动名为 `ModelWorker` 的进程，并记录日志
        _run_current_with_daemon("ModelWorker", log_file)
    else:
        # 导入 `dbgpt.model.cluster` 下的 `run_worker_manager` 函数并运行之
        from dbgpt.model.cluster import run_worker_manager

        run_worker_manager()


# 创建名为 `worker` 的点击命令，使用 `add_stop_server_options` 装饰器添加停止服务器选项
@click.command(name="worker")
@add_stop_server_options
def stop_model_worker(port: int):
    """Stop model worker"""
    # 设置 `name` 为 "ModelWorker"，如果 `port` 存在则追加端口号
    name = "ModelWorker"
    if port:
        name = f"{name}-{port}"
    # 调用 `_stop_service` 函数，停止名为 `worker` 的服务，服务名为 `name`，端口为 `port`
    _stop_service("worker", name, port=port)


# 创建名为 `apiserver` 的点击命令，使用 `EnvArgumentParser.create_click_option` 设置参数
@click.command(name="apiserver")
@EnvArgumentParser.create_click_option(ModelAPIServerParameters)
def start_apiserver(**kwargs):
    """Start apiserver"""

    # 如果 `daemon` 在参数中为真
    if kwargs["daemon"]:
        # 设置日志文件路径为 LOGDIR 下的 `model_apiserver_uvicorn.log`
        log_file = os.path.join(LOGDIR, "model_apiserver_uvicorn.log")
        # 使用 `_run_current_with_daemon` 启动名为 `ModelAPIServer` 的进程，并记录日志
        _run_current_with_daemon("ModelAPIServer", log_file)
    else:
        # 导入 `dbgpt.model.cluster` 下的 `run_apiserver` 函数并运行之
        run_apiserver()


# 创建名为 `apiserver` 的点击命令，使用 `add_stop_server_options` 装饰器添加停止服务器选项
@click.command(name="apiserver")
@add_stop_server_options
def stop_apiserver(port: int):
    """Stop apiserver"""
    # 设置 `name` 为 "ModelAPIServer"，如果 `port` 存在则追加端口号
    name = "ModelAPIServer"
    if port:
        name = f"{name}-{port}"
    # 调用 `_stop_service` 函数，停止名为 `apiserver` 的服务，服务名为 `name`，端口为 `port`
    _stop_service("apiserver", name, port=port)


# 定义名为 `_stop_all_model_server` 的函数，不接受参数，用于停止所有服务器
def _stop_all_model_server(**kwargs):
    """Stop all server"""
    # 停止名为 `worker` 的服务，服务名为 `ModelWorker`
    _stop_service("worker", "ModelWorker")
    # 停止名为 `controller` 的服务，服务名为 `ModelController`
    _stop_service("controller", "ModelController")
```