# `.\chatglm4-finetune\composite_demo\src\tools\tool_registry.py`

```py
"""
该代码是工具注册部分。通过注册工具，模型可以调用该工具。
该代码为模型提供扩展功能，使其能够通过定义的接口调用和与各种工具交互。
"""

# 导入所需的模块和类
from collections.abc import Callable  # 导入 Callable 类型用于函数类型注解
import copy  # 导入 copy 模块用于对象复制
import inspect  # 导入 inspect 模块用于获取对象的信息
import json  # 导入 json 模块用于处理 JSON 数据
from pprint import pformat  # 导入 pformat 函数用于格式化输出
import traceback  # 导入 traceback 模块用于异常跟踪
from types import GenericAlias  # 导入 GenericAlias 类型用于泛型处理
from typing import get_origin, Annotated  # 导入类型相关工具
import subprocess  # 导入 subprocess 模块用于子进程管理

from .interface import ToolObservation  # 从当前包导入 ToolObservation 类

# 从不同模块导入工具调用
from .browser import tool_call as browser  # 导入浏览器工具调用
from .cogview import tool_call as cogview  # 导入 CogView 工具调用
from .python import tool_call as python  # 导入 Python 工具调用

# 定义所有可用工具的字典
ALL_TOOLS = {
    "simple_browser": browser,  # 将浏览器工具关联到其名称
    "python": python,  # 将 Python 工具关联到其名称
    "cogview": cogview,  # 将 CogView 工具关联到其名称
}

_TOOL_HOOKS = {}  # 初始化工具钩子字典，用于存储注册的工具
_TOOL_DESCRIPTIONS = []  # 初始化工具描述列表，用于存储工具信息


def register_tool(func: Callable):
    # 获取工具的名称
    tool_name = func.__name__
    # 获取工具的描述文档并去除首尾空格
    tool_description = inspect.getdoc(func).strip()
    # 获取工具参数的签名
    python_params = inspect.signature(func).parameters
    tool_params = []  # 初始化工具参数列表
    for name, param in python_params.items():  # 遍历每个参数
        annotation = param.annotation  # 获取参数的注解
        # 检查参数是否缺少类型注解
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter `{name}` missing type annotation")
        # 检查注解类型是否为 Annotated
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")

        # 获取类型和描述、是否必需
        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__  # 确保类型为字符串
        # 检查描述是否为字符串
        if not isinstance(description, str):
            raise TypeError(f"Description for `{name}` must be a string")
        # 检查是否必需是否为布尔值
        if not isinstance(required, bool):
            raise TypeError(f"Required for `{name}` must be a bool")

        # 添加参数信息到工具参数列表
        tool_params.append(
            {
                "name": name,
                "description": description,
                "type": typ,
                "required": required,
            }
        )
    # 创建工具定义字典
    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params,
    }
    # print("[registered tool] " + pformat(tool_def))  # 可选的调试输出
    _TOOL_HOOKS[tool_name] = func  # 将工具名称与函数绑定
    _TOOL_DESCRIPTIONS.append(tool_def)  # 将工具定义添加到描述列表

    return func  # 返回注册的工具函数


def dispatch_tool(tool_name: str, code: str, session_id: str) -> list[ToolObservation]:
    # 分发预定义的工具
    if tool_name in ALL_TOOLS:
        return ALL_TOOLS[tool_name](code, session_id)  # 调用相应工具

    # 清理代码字符串
    code = code.strip().rstrip('<|observation|>').strip()

    # 分发自定义工具
    try:
        tool_params = json.loads(code)  # 尝试解析 JSON 格式的代码
    except json.JSONDecodeError as e:  # 捕获 JSON 解码错误
        err = f"Error decoding JSON: {e}"  # 创建错误信息
        return [ToolObservation("system_error", err)]  # 返回错误观察对象

    # 检查工具名称是否在已注册的工具中
    if tool_name not in _TOOL_HOOKS:
        err = f"Tool `{tool_name}` not found. Please use a provided tool."  # 错误信息
        return [ToolObservation("system_error", err)]  # 返回错误观察对象

    tool_hook = _TOOL_HOOKS[tool_name]  # 获取对应的工具钩子
    try:
        ret: str = tool_hook(**tool_params)  # 调用工具并传递参数
        return [ToolObservation(tool_name, str(ret))]  # 返回工具执行结果
    # 捕获异常，执行以下语句
    except:
        # 格式化当前异常的堆栈信息，保存到 err 变量
        err = traceback.format_exc()
        # 返回一个包含错误信息的 ToolObservation 对象的列表
        return [ToolObservation("system_error", err)]
# 获取工具的定义，返回工具描述的深拷贝列表
def get_tools() -> list[dict]:
    # 返回工具描述的深拷贝，避免修改原始数据
    return copy.deepcopy(_TOOL_DESCRIPTIONS)


# 工具定义部分


# 注册一个工具，生成随机数
@register_tool
def random_number_generator(
        # 随机生成器使用的种子，必须为整数
        seed: Annotated[int, "The random seed used by the generator", True],
        # 生成数值的范围，必须为整数元组
        range: Annotated[tuple[int, int], "The range of the generated numbers", True],
) -> int:
    """
    生成一个随机数 x，使得 range[0] <= x < range[1]
    """
    # 检查种子是否为整数
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
    # 检查范围是否为元组
    if not isinstance(range, tuple):
        raise TypeError("Range must be a tuple")
    # 检查范围的每个元素是否为整数
    if not isinstance(range[0], int) or not isinstance(range[1], int):
        raise TypeError("Range must be a tuple of integers")

    # 导入随机数模块
    import random

    # 根据种子创建随机数生成器，生成指定范围内的随机整数
    return random.Random(seed).randint(*range)


# 注册一个工具，获取天气信息
@register_tool
def get_weather(
        # 要查询的城市名称，必须为字符串
        city_name: Annotated[str, "The name of the city to be queried", True],
) -> str:
    """
    获取指定城市的当前天气
    """

    # 检查城市名称是否为字符串
    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    # 定义需要获取的天气信息的键
    key_selection = {
        "current_condition": [
            "temp_C",
            "FeelsLikeC",
            "humidity",
            "weatherDesc",
            "observation_time",
        ],
    }
    # 导入请求模块
    import requests

    try:
        # 发起请求以获取天气数据
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        # 检查请求是否成功
        resp.raise_for_status()
        # 解析返回的 JSON 数据
        resp = resp.json()
        # 构建返回的数据字典
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        # 导入追踪模块以便于错误处理
        import traceback

        # 捕获异常，返回错误信息
        ret = (
                "Error encountered while fetching weather data!\n" + traceback.format_exc()
        )

    # 返回处理后的结果
    return str(ret)


# 注册一个工具，执行 Linux 命令
@register_tool
def get_shell(
        # 要在 Linux shell 中执行的命令，必须为字符串
        query: Annotated[str, "The command should run in Linux shell", True],
) -> str:
    """
    使用 shell 执行命令
    """
    # 检查命令是否为字符串
    if not isinstance(query, str):
        raise TypeError("Command must be a string")
    try:
        # 运行命令并捕获输出和错误信息
        result = subprocess.run(
            query,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # 返回命令的标准输出
        return result.stdout
    except subprocess.CalledProcessError as e:
        # 返回命令执行的错误信息
        return e.stderr


# 如果该文件是主程序
if __name__ == "__main__":
    # 测试执行 get_shell 工具
    # print(dispatch_tool("get_shell", {"query": "pwd"}))
    # 输出获取的工具列表
    print(get_tools())
```