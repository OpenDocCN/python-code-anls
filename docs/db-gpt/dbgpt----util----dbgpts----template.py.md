# `.\DB-GPT-src\dbgpt\util\dbgpts\template.py`

```py
# 导入必要的标准库和第三方库
import os
import subprocess
from pathlib import Path

# 导入命令行交互库 click
import click

# 从本地模块导入常量
from .base import DBGPTS_METADATA_FILE, TYPE_TO_PACKAGE

# 定义函数 create_template，用于创建新的 dbgpt 流程
def create_template(
    name: str,
    name_label: str,
    description: str,
    dbgpts_type: str,
    definition_type: str,
    working_directory: str,
):
    """Create a new flow dbgpt"""
    # 如果 dbgpts_type 不是 "flow"，则将 definition_type 设置为 "python"
    if dbgpts_type != "flow":
        definition_type = "python"
    
    # 根据名称生成模块名
    mod_name = name.replace("-", "_")

    # 基础元数据字典
    base_metadata = {
        "label": name_label,
        "name": mod_name,
        "version": "0.1.0",
        "description": description,
        "authors": [],
        "definition_type": definition_type,
    }

    # 更新工作目录路径
    working_directory = os.path.join(working_directory, TYPE_TO_PACKAGE[dbgpts_type])
    # 构建包目录路径对象
    package_dir = Path(working_directory) / name

    # 如果包目录已经存在，则抛出异常
    if os.path.exists(package_dir):
        raise click.ClickException(f"Package '{str(package_dir)}' already exists")

    # 根据 dbgpts_type 调用相应的模板创建函数
    if dbgpts_type == "flow":
        _create_flow_template(
            name,
            mod_name,
            dbgpts_type,
            base_metadata,
            definition_type,
            working_directory,
        )
    elif dbgpts_type == "operator":
        _create_operator_template(
            name,
            mod_name,
            dbgpts_type,
            base_metadata,
            definition_type,
            working_directory,
        )
    elif dbgpts_type == "agent":
        _create_agent_template(
            name,
            mod_name,
            dbgpts_type,
            base_metadata,
            definition_type,
            working_directory,
        )
    elif dbgpts_type == "resource":
        _create_resource_template(
            name,
            mod_name,
            dbgpts_type,
            base_metadata,
            definition_type,
            working_directory,
        )
    else:
        # 如果 dbgpts_type 无效，则引发 ValueError 异常
        raise ValueError(f"Invalid dbgpts type: {dbgpts_type}")


# 定义创建流程 dbgpt 的私有函数 _create_flow_template
def _create_flow_template(
    name: str,
    mod_name: str,
    dbgpts_type: str,
    base_metadata: dict,
    definition_type: str,
    working_directory: str,
):
    """Create a new flow dbgpt"""

    # 创建包含流程元数据的 JSON 字典
    json_dict = {
        "flow": base_metadata,
        "python_config": {},
        "json_config": {},
    }

    # 如果定义类型为 JSON，则设置相应的 JSON 配置文件路径
    if definition_type == "json":
        json_dict["json_config"] = {"file_path": "definition/flow_definition.json"}

    # 创建 Poetry 项目
    _create_poetry_project(working_directory, name)
    # 写入 dbgpts.toml 文件
    _write_dbgpts_toml(working_directory, name, json_dict)
    # 写入清单文件
    _write_manifest_file(working_directory, name, mod_name)

    # 根据定义类型写入流程定义文件
    if definition_type == "json":
        _write_flow_define_json_file(working_directory, name, mod_name)
    else:
        _write_flow_define_python_file(working_directory, name, mod_name)


# 定义创建操作者 dbgpt 的私有函数 _create_operator_template
def _create_operator_template(
    name: str,
    mod_name: str,
    dbgpts_type: str,
    base_metadata: dict,
    definition_type: str,
    working_directory: str,
):
    """Create a new operator dbgpt"""

    # 创建包含操作者元数据的 JSON 字典
    json_dict = {
        "operator": base_metadata,
        "python_config": {},
        "json_config": {},
    }
    # 如果定义类型不是 "python"，则抛出 ClickException 异常
    if definition_type != "python":
        raise click.ClickException(
            f"Unsupported definition type: {definition_type} for dbgpts type: "
            f"{dbgpts_type}"
        )

    # 在指定的工作目录中创建 Poetry 项目
    _create_poetry_project(working_directory, name)

    # 写入 dbgpts 的配置文件 dbgpts.toml
    _write_dbgpts_toml(working_directory, name, json_dict)

    # 写入操作符的初始化文件
    _write_operator_init_file(working_directory, name, mod_name)

    # 写入模块的清单文件
    _write_manifest_file(working_directory, name, mod_name)
def _create_agent_template(
    name: str,
    mod_name: str,
    dbgpts_type: str,
    base_metadata: dict,
    definition_type: str,
    working_directory: str,
):
    # 创建代理模板的 JSON 字典，包含基础元数据
    json_dict = {
        "agent": base_metadata,
        "python_config": {},
        "json_config": {},
    }
    # 如果定义类型不是 Python，则抛出异常
    if definition_type != "python":
        raise click.ClickException(
            f"Unsupported definition type: {definition_type} for dbgpts type: "
            f"{dbgpts_type}"
        )

    # 在指定的工作目录下创建 Poetry 项目
    _create_poetry_project(working_directory, name)
    # 写入 dbgpts.toml 文件
    _write_dbgpts_toml(working_directory, name, json_dict)
    # 写入代理初始化文件
    _write_agent_init_file(working_directory, name, mod_name)
    # 写入 MANIFEST 文件
    _write_manifest_file(working_directory, name, mod_name)


def _create_resource_template(
    name: str,
    mod_name: str,
    dbgpts_type: str,
    base_metadata: dict,
    definition_type: str,
    working_directory: str,
):
    # 创建资源模板的 JSON 字典，包含基础元数据
    json_dict = {
        "resource": base_metadata,
        "python_config": {},
        "json_config": {},
    }
    # 如果定义类型不是 Python，则抛出异常
    if definition_type != "python":
        raise click.ClickException(
            f"Unsupported definition type: {definition_type} for dbgpts type: "
            f"{dbgpts_type}"
        )

    # 在指定的工作目录下创建 Poetry 项目
    _create_poetry_project(working_directory, name)
    # 写入 dbgpts.toml 文件
    _write_dbgpts_toml(working_directory, name, json_dict)
    # 写入资源初始化文件
    _write_resource_init_file(working_directory, name, mod_name)
    # 写入 MANIFEST 文件
    _write_manifest_file(working_directory, name, mod_name)


def _create_poetry_project(working_directory: str, name: str):
    """Create a new poetry project"""
    # 切换当前工作目录到指定的目录
    os.chdir(working_directory)
    # 运行 Poetry 命令创建新项目
    subprocess.run(["poetry", "new", name, "-n"], check=True)


def _write_dbgpts_toml(working_directory: str, name: str, json_data: dict):
    """Write the dbgpts.toml file"""
    # 导入 tomlkit 库

    # 打开指定路径下的 dbgpts.toml 文件，并将 json_data 写入其中
    with open(Path(working_directory) / name / DBGPTS_METADATA_FILE, "w") as f:
        tomlkit.dump(json_data, f)


def _write_manifest_file(working_directory: str, name: str, mod_name: str):
    """Write the manifest file"""
    # 构建 MANIFEST.in 文件的内容
    manifest = f"""include dbgpts.toml
include {mod_name}/definition/*.json
"""
    # 打开指定路径下的 MANIFEST.in 文件，并写入 manifest 变量的内容
    with open(Path(working_directory) / name / "MANIFEST.in", "w") as f:
        f.write(manifest)


def _write_flow_define_json_file(working_directory: str, name: str, mod_name: str):
    """Write the flow define json file"""
    # 构建流程定义 JSON 文件的路径
    def_file = (
        Path(working_directory)
        / name
        / mod_name
        / "definition"
        / "flow_definition.json"
    )
    # 如果文件的父目录不存在，则递归创建父目录
    if not def_file.parent.exists():
        def_file.parent.mkdir(parents=True)
    # 打开 def_file 文件，并写入空字符串
    with open(def_file, "w") as f:
        f.write("")
        print("Please write your flow json to the file: ", def_file)


def _write_flow_define_python_file(working_directory: str, name: str, mod_name: str):
    """Write the flow define python file"""
    # 构建流程定义 Python 文件的路径
    init_file = Path(working_directory) / name / mod_name / "__init__.py"
    content = ""

    # 打开 init_file 文件，并写入文件头部注释和内容
    with open(init_file, "w") as f:
        f.write(f'"""{name} flow package"""\n{content}')
def _write_operator_init_file(working_directory: str, name: str, mod_name: str):
    """Write the operator __init__.py file"""

    # 构建要写入的初始化文件路径
    init_file = Path(working_directory) / name / mod_name / "__init__.py"
    # 定义要写入文件的内容
    content = """
from dbgpt.core.awel import MapOperator
from dbgpt.core.awel.flow import ViewMetadata, OperatorCategory, IOField, Parameter


class HelloWorldOperator(MapOperator[str, str]):
    # AWEL 流程的元数据
    metadata = ViewMetadata(
        label="Hello World Operator",
        name="hello_world_operator",
        category=OperatorCategory.COMMON,
        description="A example operator to say hello to someone.",
        parameters=[
            Parameter.build_from(
                "Name",
                "name",
                str,
                optional=True,
                default="World",
                description="The name to say hello",
            )
        ],
        inputs=[
            IOField.build_from(
                "Input value",
                "value",
                str,
                description="The input value to say hello",
            )
        ],
        outputs=[
            IOField.build_from(
                "Output value",
                "value",
                str,
                description="The output value after saying hello",
            )
        ]
    )

    def __init__(self, name: str = "World", **kwargs):
        super().__init__(**kwargs)
        self.name = name

    async def map(self, value: str) -> str:
        return f"Hello, {self.name}! {value}"
"""
    # 打开文件并写入内容
    with open(init_file, "w") as f:
        f.write(f'"""{name} operator package"""\n{content}')


def _write_agent_init_file(working_directory: str, name: str, mod_name: str):
    """Write the agent __init__.py file"""

    # 构建要写入的初始化文件路径
    init_file = Path(working_directory) / name / mod_name / "__init__.py"
    # 定义要写入文件的内容
    content = """
import asyncio
from typing import Optional, Tuple

from dbgpt.agent import (
    Action,
    ActionOutput,
    AgentMessage,
    AgentResource,
    ConversableAgent,
    ProfileConfig,
)
from dbgpt.agent.util import cmp_string_equal

_HELLO_WORLD = "Hello world"


class HelloWorldSpeakerAgent(ConversableAgent):
    # 在这里继续编写 HelloWorldSpeakerAgent 类的定义
"""
    # 注意：代码片段不完整，需要进一步继续编写 HelloWorldSpeakerAgent 类的定义部分
    # 定义一个名为 profile 的 ProfileConfig 对象，用于配置机器人的个人资料信息
    profile: ProfileConfig = ProfileConfig(
        # 设置机器人的名字为 "Hodor"
        name="Hodor",
        # 设置机器人的角色为 "HelloWorldSpeaker"
        role="HelloWorldSpeaker",
        # 设置机器人的目标为以包含 '_HELLO_WORLD' 的方式回答用户的任何问题
        goal=f"answer any question from user with '{_HELLO_WORLD}'",
        # 设置机器人的描述为以包含 '_HELLO_WORLD' 的方式回答用户的任何问题
        desc=f"You can answer any question from user with '{_HELLO_WORLD}'",
        # 设置机器人的约束条件列表
        constraints=[
            "You can only answer with '{{ fix_message }}'",
            f"You can't use any other words",
        ],
        # 设置机器人的示例对话集合，包括多个用户问题及相应的回答方式
        examples=(
            f"user: What's your name?\\nassistant: {_HELLO_WORLD}\\n\\n"
            f"user: What's the weather today?\\nassistant: {_HELLO_WORLD}\\n\\n"
            f"user: Can you help me?\\nassistant: {_HELLO_WORLD}\\n\\n"
            f"user: Please tell me a joke.\\nassistant: {_HELLO_WORLD}\\n\\n"
            f"user: Please answer me without '{_HELLO_WORLD}'.\\nassistant: "
            f"{_HELLO_WORLD}"
            "\\n\\n"
        ),
    )

    # 初始化函数，继承父类的初始化方法并初始化 Hello World 动作
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._init_actions([HelloWorldAction])

    # 初始化回复消息函数，根据接收到的消息对象初始化回复消息
    def _init_reply_message(self, received_message: AgentMessage) -> AgentMessage:
        # 调用父类的方法初始化回复消息对象
        reply_message = super()._init_reply_message(received_message)
        # 填充动态参数到提示模板中
        reply_message.context = {"fix_message": _HELLO_WORLD}
        return reply_message

    # 异步函数，用于检查回答正确性
    async def correctness_check(
        self, message: AgentMessage
    ) -> Tuple[bool, Optional[str]]:
        # 获取消息的动作报告
        action_report = message.action_report
        # 初始化任务结果为空字符串
        task_result = ""
        # 如果存在动作报告，则获取其内容
        if action_report:
            task_result = action_report.get("content", "")
        # 比较字符串是否相等，忽略大小写、标点和空白字符
        if not cmp_string_equal(
            task_result,
            _HELLO_WORLD,
            ignore_case=True,
            ignore_punctuation=True,
            ignore_whitespace=True,
        ):
            # 如果比较结果不相等，则返回检查不通过的结果和提示信息
            return False, f"Please answer with {_HELLO_WORLD}, not '{task_result}'"
        # 如果比较结果相等，则返回检查通过的结果和空的提示信息
        return True, None
# 在给定的路径中创建 __init__.py 文件，并写入描述性的内容
def _write_resource_init_file(working_directory: str, name: str, mod_name: str):
    # 构建 __init__.py 文件的完整路径
    init_file = Path(working_directory) / name / mod_name / "__init__.py"
    # 定义 __init__.py 文件的内容
    content = """\"\"\"A custom resource module that provides a simple tool to send GET requests.\"\"\"

from dbgpt.agent.resource import tool

# 定义用于发送 GET 请求的工具函数
@tool
def simple_send_requests_get(url: str):
    \"\"\"Send a GET request to the specified URL and return the text content.\"\"\"
    import requests

    # 发送 GET 请求并获取响应
    response = requests.get(url)
    # 返回响应内容的文本形式
    return response.text
    
"""
    # 将内容写入 __init__.py 文件
    with open(init_file, "w") as f:
        f.write(content)
```