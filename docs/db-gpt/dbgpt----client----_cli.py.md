# `.\DB-GPT-src\dbgpt\client\_cli.py`

```py
"""CLI for DB-GPT client."""

# 引入必要的库
import asyncio
import functools
import json
import time
import uuid
from typing import Any, AsyncIterator, Callable, Dict, Tuple, cast

import click

# 引入自定义模块
from dbgpt.component import SystemApp
from dbgpt.core.awel import DAG, BaseOperator, DAGVar
from dbgpt.core.awel.dag.dag_manager import DAGMetadata, _parse_metadata
from dbgpt.core.awel.flow.flow_factory import FlowFactory
from dbgpt.util import get_or_create_event_loop
from dbgpt.util.console import CliLogger
from dbgpt.util.i18n_utils import _

from .client import Client
from .flow import list_flow
from .flow import run_flow_cmd as client_run_flow_cmd

# 创建 CLILogger 实例
cl = CliLogger()

# 定义全局变量
_LOCAL_MODE: bool | None = False
_FILE_PATH: str | None = None

# 创建命令组
@click.group()
@click.option(
    "--local",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Whether use local mode(run local AWEL file)",
)
@click.option(
    "-f",
    "--file",
    type=str,
    default=None,
    required=False,
    help=_("The path of the AWEL flow"),
)
def flow(local: bool = False, file: str | None = None):
    """Run a AWEL flow."""
    global _LOCAL_MODE, _FILE_PATH
    _LOCAL_MODE = local
    _FILE_PATH = file

# 添加基础流程选项到命令
def add_base_flow_options(func):
    """Add base flow options to the command."""

    @click.option(
        "-n",
        "--name",
        type=str,
        default=None,
        required=False,
        help=_("The name of the AWEL flow"),
    )
    @click.option(
        "--uid",
        type=str,
        default=None,
        required=False,
        help=_("The uid of the AWEL flow"),
    )
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return _wrapper

# 添加聊天选项到命令
def add_chat_options(func):
    """Add chat options to the command."""

    @click.option(
        "-m",
        "--messages",
        type=str,
        default=None,
        required=False,
        help=_("The messages to run AWEL flow"),
    )
    @click.option(
        "--model",
        type=str,
        default=None,
        required=False,
        help=_("The model name of AWEL flow"),
    )
    @click.option(
        "-s",
        "--stream",
        type=bool,
        default=False,
        required=False,
        is_flag=True,
        help=_("Whether use stream mode to run AWEL flow"),
    )
    @click.option(
        "-t",
        "--temperature",
        type=float,
        default=None,
        required=False,
        help=_("The temperature to run AWEL flow"),
    )
    @click.option(
        "--max_new_tokens",
        type=int,
        default=None,
        required=False,
        help=_("The max new tokens to run AWEL flow"),
    )
    @click.option(
        "--conv_uid",
        type=str,
        default=None,
        required=False,
        help=_("The conversation id of the AWEL flow"),
    )
    # 使用 click 库的装饰器 @click.option 添加命令行选项 -d 或 --data，用于接收 JSON 数据以运行 AWEL 流程；如果设置，将覆盖其他选项
    @click.option(
        "-d",
        "--data",
        type=str,
        default=None,
        required=False,
        help=_("The json data to run AWEL flow, if set, will overwrite other options"),
    )
    # 使用 click 库的装饰器 @click.option 添加命令行选项 -e 或 --extra，用于接收额外的 JSON 数据以运行 AWEL 流程
    @click.option(
        "-e",
        "--extra",
        type=str,
        default=None,
        required=False,
        help=_("The extra json data to run AWEL flow."),
    )
    # 使用 click 库的装饰器 @click.option 添加命令行选项 -i 或 --interactive，接收布尔值类型的参数，用于确定是否使用交互模式运行 AWEL 流程
    @click.option(
        "-i",
        "--interactive",
        type=bool,
        default=False,
        required=False,
        is_flag=True,
        help=_("Whether use interactive mode to run AWEL flow"),
    )
    # 使用 functools 模块的 wraps 装饰器，将 func 函数的元信息（如函数名、参数列表等）复制到 _wrapper 函数中
    @functools.wraps(func)
    # 定义 _wrapper 函数，作为装饰器的包装函数，接受任意位置和关键字参数，并将它们传递给被装饰的 func 函数
    def _wrapper(*args, **kwargs):
        # 调用被装饰的 func 函数，并将接收到的参数传递给它，然后返回其返回值
        return func(*args, **kwargs)
    
    # 返回 _wrapper 函数，作为装饰器的结果
    return _wrapper
# 定义一个命令装饰器，指定该函数为 AWEL 流的聊天模式运行函数
@flow.command(name="chat")
# 添加基础流选项的装饰器
@add_base_flow_options
# 添加聊天选项的装饰器
@add_chat_options
# 函数签名，接受多个参数，包括名称、用户ID、数据、交互标志等
def run_flow_chat(name: str, uid: str, data: str, interactive: bool, **kwargs):
    """Run a AWEL flow."""
    # 解析聊天模式下的 JSON 数据
    json_data = _parse_chat_json_data(data, **kwargs)
    # 检查 JSON 数据中是否包含流标志并将其转换为布尔型
    stream = "stream" in json_data and str(json_data["stream"]).lower() in ["true", "1"]
    # 获取或创建事件循环
    loop = get_or_create_event_loop()
    # 如果处于本地模式
    if _LOCAL_MODE:
        # 在本地环境中运行聊天模式的流程
        _run_flow_chat_local(loop, name, interactive, json_data, stream)
        # 返回函数，结束执行
        return

    # 创建客户端对象
    client = Client()

    # 将名称中的连字符替换为下划线，适应 AWEL 流程存储的 Python 模块名称规范
    new_name = name.replace("-", "_")
    # 在事件循环中运行并等待流程列表的返回结果
    res = loop.run_until_complete(list_flow(client, new_name, uid))

    # 如果没有找到结果
    if not res:
        # 打印错误信息并退出，指定退出码为1
        cl.error("Flow not found with the given name or uid", exit_code=1)
    # 如果返回结果超过一个
    if len(res) > 1:
        # 打印错误信息并退出，指定退出码为1
        cl.error("More than one flow found", exit_code=1)
    
    # 获取第一个流程对象
    flow = res[0]
    # 设置 JSON 数据中的聊天参数为流程的 UID
    json_data["chat_param"] = flow.uid
    # 设置 JSON 数据中的聊天模式为 chat_flow
    json_data["chat_mode"] = "chat_flow"
    # 如果流标志为真，运行流程的聊天模式流程
    if stream:
        _run_flow_chat_stream(loop, client, interactive, json_data)
    # 否则，运行普通的聊天模式流程
    else:
        _run_flow_chat(loop, client, interactive, json_data)


# 定义一个命令装饰器，指定该函数为 AWEL 流的命令行模式运行函数
@flow.command(name="cmd")
# 添加基础流选项的装饰器
@add_base_flow_options
# 添加命令行选项的装饰器，包括 JSON 数据和输出键选项
@click.option(
    "-d",
    "--data",
    type=str,
    default=None,
    required=False,
    help=_("The json data to run AWEL flow, if set, will overwrite other options"),
)
@click.option(
    "--output_key",
    type=str,
    default=None,
    required=False,
    help=_(
        "The output key of the AWEL flow, if set, it will try to get the output by the "
        "key"
    ),
)
# 函数签名，接受名称、用户ID、可选的 JSON 数据和输出键参数
def run_flow_cmd(
    name: str, uid: str, data: str | None = None, output_key: str | None = None
):
    """Run a AWEL flow with command mode."""
    # 解析命令行模式下的 JSON 数据
    json_data = _parse_json_data(data)
    # 获取或创建事件循环
    loop = get_or_create_event_loop()

    # 如果处于本地模式
    if _LOCAL_MODE:
        # 在本地环境中运行命令行模式的流程
        _run_flow_cmd_local(loop, name, json_data, output_key)
    else:
        # 在远程环境中运行命令行模式的流程
        _run_flow_cmd(loop, name, uid, json_data, output_key)


# 定义一个私有函数，在本地环境中运行命令行模式的流程
def _run_flow_cmd_local(
    loop: asyncio.BaseEventLoop,
    name: str,
    data: Dict[str, Any] | None = None,
    output_key: str | None = None,
):
    # 导入用于安全聊天流程和 DAG 任务的工具模块
    from dbgpt.core.awel.util.chat_util import safe_chat_stream_with_dag_task

    # 解析和检查本地 DAG 的最终节点、DAG、DAG 元数据和调用体
    end_node, dag, dag_metadata, call_body = _parse_and_check_local_dag(
        name, _FILE_PATH, data
    )
    # 定义一个异步函数 `_streaming_call`，用于执行异步流式调用
    async def _streaming_call():
        # 记录函数开始时间
        start_time = time.time()
        try:
            # 输出调试信息，标记流程开始
            cl.debug("[~info] Flow started")
            # 输出调试信息，显示JSON数据的内容，确保非ASCII字符不转义
            cl.debug(f"[~info] JSON data: {json.dumps(data, ensure_ascii=False)}")
            # 输出调试信息，标记命令输出开始
            cl.debug("Command output: ")
            # 异步迭代调用 `safe_chat_stream_with_dag_task` 函数返回的流
            async for out in safe_chat_stream_with_dag_task(
                end_node, call_body, incremental=True, covert_to_str=True
            ):
                # 如果调用结果不成功，输出错误信息
                if not out.success:
                    cl.error(out.text)
                else:
                    # 如果调用结果成功，输出调用结果文本内容（不换行）
                    cl.print(out.text, end="")
        except Exception as e:
            # 捕获异常情况，输出错误信息并设置退出码为1
            cl.error(f"Failed to run flow: {e}", exit_code=1)
        finally:
            # 计算流程执行时间并四舍五入保留两位小数，输出成功信息和流程耗时
            time_cost = round(time.time() - start_time, 2)
            cl.success(f"\n:tada: Flow finished, timecost: {time_cost} s")

    # 使用事件循环执行 `_streaming_call` 函数
    loop.run_until_complete(_streaming_call())
# 定义一个异步函数，用于运行流程命令
async def _client_run_cmd():
    # 输出调试信息，表示流程开始
    cl.debug("[~info] Flow started")
    # 输出调试信息，显示传入的 JSON 数据
    cl.debug(f"[~info] JSON data: {json.dumps(json_data, ensure_ascii=False)}")
    # 输出调试信息，标识命令输出即将展示
    cl.debug("Command output: ")
    # 记录当前时间，用于计算命令执行耗时
    start_time = time.time()
    # 将命令名称中的 "-" 替换为 "_"，因为 AWEL 流程现在存储 Python 模块名称
    new_name = name.replace("-", "_")
    try:
        # 调用客户端运行流程命令的函数，传入相应参数和回调函数
        await client_run_flow_cmd(
            client,
            new_name,
            uid,
            json_data,
            non_streaming_callback=_non_streaming_callback,
            streaming_callback=_streaming_callback,
        )
    except Exception as e:
        # 若执行命令过程中出现异常，记录错误信息并退出
        cl.error(f"Failed to run flow: {e}", exit_code=1)
    finally:
        # 计算并记录命令执行耗时，以秒为单位
        time_cost = round(time.time() - start_time, 2)
        cl.success(f"\n:tada: Flow finished, timecost: {time_cost} s")

# 运行事件循环，执行 _client_run_cmd() 函数
loop.run_until_complete(_client_run_cmd())
    # 如果触发节点列表非空
    if trigger_nodes:
        # 如果触发节点列表的长度大于1
        if len(trigger_nodes) > 1:
            # 报错：流程中找到多个触发节点
            cl.error("More than one trigger nodes found in the flow", exit_code=1)
        
        # 取第一个触发节点作为触发器
        trigger = trigger_nodes[0]
        
        # 如果触发器的类型是 HttpTrigger
        if isinstance(trigger, HttpTrigger):
            # 将触发器强制转换为 HttpTrigger 类型
            http_trigger = trigger
            
            # 如果 HttpTrigger 对象中的请求体 (_req_body) 存在，并且提供了数据
            if http_trigger._req_body and data:
                # 调用触发器的请求体方法，传入数据，并获取返回值
                call_body = http_trigger._req_body(**data)
        
        # 如果触发器类型不是 HttpTrigger
        else:
            # 报错：不支持的触发器类型
            cl.error("Unsupported trigger type", exit_code=1)
    
    # 返回结束节点、有向无环图（DAG）、DAG 元数据和调用体（如果存在）
    return end_node, dag, dag_metadata, call_body
# 解析本地 DAG，返回解析后的 DAG 对象和元数据
def _parse_local_dag(name: str, filepath: str | None = None) -> Tuple[DAG, DAGMetadata]:
    # 创建系统应用对象
    system_app = SystemApp()
    # 设置当前系统应用
    DAGVar.set_current_system_app(system_app)

    if not filepath:
        # 从安装的包中加载 DAG（dbgpts）
        from dbgpt.util.dbgpts.loader import (
            _flow_package_to_flow_panel,
            _load_flow_package_from_path,
        )
        # 将流程包转换为流程面板
        flow_panel = _flow_package_to_flow_panel(_load_flow_package_from_path(name))
        # 根据流程面板类型选择构建工厂或直接使用流程 DAG
        if flow_panel.define_type == "json":
            factory = FlowFactory()
            factory.pre_load_requirements(flow_panel)
            dag = factory.build(flow_panel)
        else:
            dag = flow_panel.flow_dag
        # 返回 DAG 对象和解析后的元数据
        return dag, _parse_metadata(dag)
    else:
        # 从文件路径加载 DAG
        from dbgpt.core.awel.dag.loader import _process_file
        dags = _process_file(filepath)
        if not dags:
            cl.error("No DAG found in the file", exit_code=1)
        if len(dags) > 1:
            # 如果找到多个 DAG，则根据名称过滤
            dags = [dag for dag in dags if dag.dag_id == name]
            if len(dags) > 1:
                cl.error("More than one DAG found in the file", exit_code=1)
        if not dags:
            cl.error("No DAG found with the given name", exit_code=1)
        # 返回第一个 DAG 对象和解析后的元数据
        return dags[0], _parse_metadata(dags[0])


# 解析聊天 JSON 数据并处理额外的关键字参数
def _parse_chat_json_data(data: str, **kwargs):
    json_data = {}
    if data:
        try:
            json_data = json.loads(data)
        except Exception as e:
            cl.error(f"Invalid JSON data: {data}, {e}", exit_code=1)
    if "extra" in kwargs and kwargs["extra"]:
        try:
            extra = json.loads(kwargs["extra"])
            kwargs["extra"] = extra
        except Exception as e:
            cl.error(f"Invalid extra JSON data: {kwargs['extra']}, {e}", exit_code=1)
    # 将关键字参数合并到 JSON 数据中
    for k, v in kwargs.items():
        if v is not None and k not in json_data:
            json_data[k] = v
    # 如果 JSON 数据中没有模型字段，则设置默认值
    if "model" not in json_data:
        json_data["model"] = "__empty__model__"
    return json_data


# 解析 JSON 数据，返回字典或 None
def _parse_json_data(data: str | None) -> Dict[str, Any] | None:
    if not data:
        return None
    try:
        return json.loads(data)
    except Exception as e:
        cl.error(f"Invalid JSON data: {data}, {e}", exit_code=1)
        # 不应该执行到这里
        return None


# 运行本地流程聊天任务，与 DAG 任务相关联
def _run_flow_chat_local(
    loop: asyncio.BaseEventLoop,
    name: str,
    interactive: bool,
    json_data: Dict[str, Any],
    stream: bool,
):
    from dbgpt.core.awel.util.chat_util import (
        parse_single_output,
        safe_chat_stream_with_dag_task,
    )

    # 解析本地 DAG，获取 DAG 对象和元数据
    dag, dag_metadata = _parse_local_dag(name, _FILE_PATH)
    # 定义一个异步函数 _streaming_call，接受一个字典类型的参数 _call_body
    async def _streaming_call(_call_body: Dict[str, Any]):
        # 声明在函数作用域中要使用的非局部变量 dag 和 dag_metadata
        nonlocal dag, dag_metadata

        # 调用 _check_local_dag 函数，检查本地的 DAG（有向无环图），获取处理后的结果
        end_node, dag, dag_metadata, handled_call_body = _check_local_dag(
            dag, dag_metadata, _call_body
        )
        
        # 通过 safe_chat_stream_with_dag_task 安全地与 DAG 任务进行交互流处理
        async for out in safe_chat_stream_with_dag_task(
            end_node, handled_call_body, incremental=True, covert_to_str=True
        ):
            # 如果交互结果不成功，记录错误并抛出异常
            if not out.success:
                cl.error(f"Error: {out.text}")
                raise Exception(out.text)
            else:
                # 如果成功，生成输出文本
                yield out.text

    # 定义一个异步函数 _call，接受一个字典类型的参数 _call_body
    async def _call(_call_body: Dict[str, Any]):
        # 声明在函数作用域中要使用的非局部变量 dag 和 dag_metadata
        nonlocal dag, dag_metadata

        # 调用 _check_local_dag 函数，检查本地的 DAG（有向无环图），获取处理后的结果
        end_node, dag, dag_metadata, handled_call_body = _check_local_dag(
            dag, dag_metadata, _call_body
        )
        
        # 调用 end_node 对象的 call 方法，使用处理后的请求体进行调用
        res = await end_node.call(handled_call_body)
        
        # 解析单个输出结果，转换为字符串形式（非 SSE），并获取解析结果
        parsed_res = parse_single_output(res, is_sse=False, covert_to_str=True)
        
        # 如果解析结果不成功，抛出异常
        if not parsed_res.success:
            raise Exception(parsed_res.text)
        
        # 返回解析后的文本结果
        return parsed_res.text

    # 如果 stream 为真，则运行 _chat_stream 函数，传入 _streaming_call 函数作为参数
    if stream:
        loop.run_until_complete(_chat_stream(_streaming_call, interactive, json_data))
    # 如果 stream 为假，则运行 _chat 函数，传入 _call 函数作为参数
    else:
        loop.run_until_complete(_chat(_call, interactive, json_data))
# 定义一个异步函数，用于处理聊天流，接受一个字典参数_call_body
async def _streaming_call(_call_body: Dict[str, Any]):
    # 使用客户端对象的 chat_stream 方法进行异步迭代
    async for out in client.chat_stream(**_call_body):
        # 如果输出中有选择项
        if out.choices:
            # 获取第一个选择项的文本内容
            text = out.choices[0].delta.content
            # 如果文本内容存在，则生成该文本
            if text:
                yield text

# 运行直到完成 chat_stream 流处理的循环
loop.run_until_complete(_chat_stream(_streaming_call, interactive, json_data))


# 定义一个异步函数，用于处理聊天，接受一个字典参数_call_body
async def _call(_call_body: Dict[str, Any]):
    # 使用客户端对象的 chat 方法进行异步调用
    res = await client.chat(**_call_body)
    # 如果返回结果中有选择项
    if res.choices:
        # 获取第一个选择项的消息内容
        text = res.choices[0].message.content
        # 返回消息内容
        return text

# 运行直到完成 chat 函数调用的处理
loop.run_until_complete(_chat(_call, interactive, json_data))


# 定义一个异步函数，用于处理聊天流，接受一个流函数参数和JSON数据
async def _chat_stream(
    streaming_func: Callable[[Dict[str, Any]], AsyncIterator[str]],
    interactive: bool,
    json_data: Dict[str, Any],
):
    # 获取用户输入的消息，默认为空字符串
    user_input = json_data.get("messages", "")
    # 如果 JSON 数据中没有会话 UID 并且是交互模式
    if "conv_uid" not in json_data and interactive:
        # 创建一个新的会话 UID
        json_data["conv_uid"] = str(uuid.uuid4())
    # 标记是否为第一条消息
    first_message = True
    while True:
        try:
            # 如果是交互模式并且用户输入为空
            if interactive and not user_input:
                # 提示用户输入 'exit' 或 'quit' 来退出
                cl.print("Type 'exit' or 'quit' to exit.")
                # 当用户输入为空时，持续询问
                while not user_input:
                    user_input = cl.ask("You")
            # 如果用户输入为 'exit', 'quit' 或 'q' 中的任意一个
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            # 记录开始处理时间
            start_time = time.time()
            # 将用户输入的消息放入 JSON 数据中
            json_data["messages"] = user_input
            # 如果是第一条消息，显示用户输入内容
            if first_message:
                cl.info("You: " + user_input)
            # 输出调试信息，指示聊天流开始
            cl.debug("[~info] Chat stream started")
            # 输出 JSON 数据的调试信息
            cl.debug(f"[~info] JSON data: {json.dumps(json_data, ensure_ascii=False)}")
            # 打印 'Bot: ' 提示
            cl.print("Bot: ")
            # 异步迭代流函数处理返回的文本流
            async for text in streaming_func(json_data):
                # 如果文本不为空，将文本内容打印出来
                if text:
                    cl.print(text, end="")
            # 记录结束处理时间
            end_time = time.time()
            # 计算处理时间消耗
            time_cost = round(end_time - start_time, 2)
            # 输出成功信息，显示聊天流处理完成和消耗的时间
            cl.success(f"\n:tada: Chat stream finished, timecost: {time_cost} s")
        # 捕获可能发生的异常
        except Exception as e:
            cl.error(f"Chat stream failed: {e}", exit_code=1)
        # 无论是否发生异常，确保首条消息标记设为 False
        finally:
            first_message = False
            # 如果是交互模式，重置用户输入为空字符串，以便下一次循环
            if interactive:
                user_input = ""
            # 如果不是交互模式，跳出循环
            else:
                break


# 定义一个异步函数，用于处理聊天，接受一个函数参数和JSON数据
async def _chat(
    func: Callable[[Dict[str, Any]], Any],
    interactive: bool,
    json_data: Dict[str, Any],
):
    # 获取用户输入的消息，默认为空字符串
    user_input = json_data.get("messages", "")
    # 如果 JSON 数据中没有会话 UID 并且是交互模式
    if "conv_uid" not in json_data and interactive:
        # 创建一个新的会话 UID
        json_data["conv_uid"] = str(uuid.uuid4())
    # 标记是否为第一条消息
    first_message = True
    # 进入无限循环，等待用户输入或者退出命令
    while True:
        try:
            # 如果是交互模式并且没有用户输入，则提示用户输入退出命令
            if interactive and not user_input:
                cl.print("Type 'exit' or 'quit' to exit.")
                # 循环等待用户输入非空字符串
                while not user_input:
                    user_input = cl.ask("You")
            
            # 如果用户输入的内容是退出命令，则跳出循环
            if user_input.lower() in ["exit", "quit", "q"]:
                break
            
            # 记录开始时间
            start_time = time.time()
            
            # 将用户输入的消息放入 JSON 数据中的消息字段
            json_data["messages"] = user_input
            
            # 如果是第一条消息，则记录用户输入日志
            if first_message:
                cl.info("You: " + user_input)

            # 调试信息：聊天开始
            cl.debug("[~info] Chat started")
            
            # 调试信息：输出 JSON 数据内容
            cl.debug(f"[~info] JSON data: {json.dumps(json_data, ensure_ascii=False)}")
            
            # 调用异步函数处理 JSON 数据，获取返回结果
            res = await func(json_data)
            
            # 打印机器人的回复
            cl.print("Bot: ")
            if res:
                cl.markdown(res)
            
            # 计算本次聊天耗时
            time_cost = round(time.time() - start_time, 2)
            
            # 成功消息：聊天流程结束，显示耗时
            cl.success(f"\n:tada: Chat stream finished, timecost: {time_cost} s")
        
        # 捕获所有异常
        except Exception as e:
            import traceback
            
            # 获取异常的详细信息并记录日志
            messages = traceback.format_exc()
            cl.error(f"Chat failed: {e}\n, error detail: {messages}", exit_code=1)
        
        # 无论是否发生异常，最终执行的代码块
        finally:
            # 将第一条消息标志设为 False，表示不再是第一条消息
            first_message = False
            
            # 如果是交互模式，重置用户输入为空字符串，继续循环等待下一次输入
            if interactive:
                user_input = ""
            
            # 如果不是交互模式，跳出循环结束程序
            else:
                break
```