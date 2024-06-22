# `.\transformers\tools\agents.py`

```py
#!/usr/bin/env python
# coding=utf-8

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归The HuggingFace Inc.团队所有

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 在遵守许可证的情况下，可以使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 许可证信息

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律有要求或书面同意，否则在许可下分发的软件将按"按原样"基础分发，没有任何明示或暗示的保证或条件，查看许可证以获取特定语言管理权限和限制

# 导入模块
import importlib.util
import json
import os
import time
from dataclasses import dataclass
from typing import Dict

# 导入requests模块
import requests
# 导入自定义模块
from huggingface_hub import HfFolder, hf_hub_download, list_spaces
# 导入自定义模块
from ..models.auto import AutoTokenizer
# 导入自定义模块
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging
# 导入自定义模块
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote
# 导入自定义模块
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt
# 导入自定义模块
from .python_interpreter import evaluate

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果openai可用
if is_openai_available():
    # 导入openai模块
    import openai

# 如果torch可用
if is_torch_available():
    # 导入相关模块
    from ..generation import StoppingCriteria, StoppingCriteriaList
    from ..models.auto import AutoModelForCausalLM
# 如果torch不可用
else:
    # 定义对象
    StoppingCriteria = object

# 初始化标志
_tools_are_initialized = False

# 基础python工具
BASE_PYTHON_TOOLS = {
    "print": print,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
}

# 预处理工具
@dataclass
class PreTool:
    task: str
    description: str
    repo_id: str

# 默认的Huggingface工具
HUGGINGFACE_DEFAULT_TOOLS = {}

# 从hub获取的默认HuggingFace工具
HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB = [
    "image-transformation",
    "text-download",
    "text-to-image",
    "text-to-video",
]

# 获取远程工具
def get_remote_tools(organization="huggingface-tools"):
    # 如果处于离线模式
    if is_offline_mode():
        # 输出日志信息
        logger.info("You are in offline mode, so remote tools are not available.")
        # 返回空字典
        return {}

    # 获取空间列表
    spaces = list_spaces(author=organization)
    # 初始化工具字典
    tools = {}
    # 遍历空间信息
    for space_info in spaces:
        # 获取repo_id
        repo_id = space_info.id
        # 解析配置文件路径
        resolved_config_file = hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space")
        # 打开配置文件，读取配置信息
        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)

        # 获取任务名
        task = repo_id.split("/")[-1]
        # 添加工具信息到工具字典
        tools[config["name"]] = PreTool(task=task, description=config["description"], repo_id=repo_id)

    # 返回工具字典
    return tools

# 设置默认工具
def _setup_default_tools():
    # 声明全局变量
    global HUGGINGFACE_DEFAULT_TOOLS
    global _tools_are_initialized

    # 如果工具已初始化，直接返回
    if _tools_are_initialized:
        return

    # 导入模块
    main_module = importlib.import_module("transformers")
    tools_module = main_module.tools

    # 获取远程工具
    remote_tools = get_remote_tools()
    # 遍历任务映射字典中的每个任务名和工具类名
    for task_name, tool_class_name in TASK_MAPPING.items():
        # 从工具模块中获取工具类对象
        tool_class = getattr(tools_module, tool_class_name)
        # 获取工具描述
        description = tool_class.description
        # 将工具类的名称映射到默认工具字典中，并设置任务名、描述为空、repo_id为空
        HUGGINGFACE_DEFAULT_TOOLS[tool_class.name] = PreTool(task=task_name, description=description, repo_id=None)

    # 如果不是离线模式
    if not is_offline_mode():
        # 遍历来自 Hugging Face Hub 的默认工具列表
        for task_name in HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB:
            found = False
            # 遍历远程工具字典的工具名和工具对象
            for tool_name, tool in remote_tools.items():
                # 如果工具的任务与当前遍历的任务名相同
                if tool.task == task_name:
                    # 将该工具添加到默认工具字典中
                    HUGGINGFACE_DEFAULT_TOOLS[tool_name] = tool
                    found = True
                    break

            # 如果未找到任务实现
            if not found:
                raise ValueError(f"{task_name} is not implemented on the Hub.")

    # 标记工具初始化完成
    _tools_are_initialized = True
# 解析工具箱中的工具，并返回解析后的工具字典
def resolve_tools(code, toolbox, remote=False, cached_tools=None):
    # 如果没有缓存的工具，则创建一个副本
    if cached_tools is None:
        resolved_tools = BASE_PYTHON_TOOLS.copy()
    else:
        resolved_tools = cached_tools
    # 遍历工具箱中的工具
    for name, tool in toolbox.items():
        # 如果工具不在代码中或者已经在解析后的工具中，则继续
        if name not in code or name in resolved_tools:
            continue

        # 如果工具是 Tool 类型，则直接加入解析后的工具字典
        if isinstance(tool, Tool):
            resolved_tools[name] = tool
        else:
            # 否则根据任务或存储库 ID 加载工具
            task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
            _remote = remote and supports_remote(task_or_repo_id)
            resolved_tools[name] = load_tool(task_or_repo_id, remote=_remote)

    # 返回解析后的工具字典
    return resolved_tools


# 获取工具创建的代码，并返回代码字符串
def get_tool_creation_code(code, toolbox, remote=False):
    # 初始化代码行
    code_lines = ["from transformers import load_tool", ""]
    # 遍历工具箱中的工具
    for name, tool in toolbox.items():
        # 如果工具不在代码中或者是 Tool 类型，则继续
        if name not in code or isinstance(tool, Tool):
            continue

        # 根据任务或存储库 ID 加载工具，并生成代码行
        task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
        line = f'{name} = load_tool("{task_or_repo_id}"'
        if remote:
            line += ", remote=True"
        line += ")"
        code_lines.append(line)

    # 返回拼接后的代码字符串
    return "\n".join(code_lines) + "\n"


# 清理用于聊天的代码
def clean_code_for_chat(result):
    lines = result.split("\n")
    idx = 0
    # 找到代码块开始的索引
    while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
        idx += 1
    explanation = "\n".join(lines[:idx]).strip()
    # 如果没有找到代码块，则返回解释部分和空代码部分
    if idx == len(lines):
        return explanation, None

    idx += 1
    start_idx = idx
    # 找到代码块结束的索引
    while not lines[idx].lstrip().startswith("```py"):
        idx += 1
    code = "\n".join(lines[start_idx:idx]).strip()

    # 返回解释部分和代码部分
    return explanation, code


# 清理用于运行的代码
def clean_code_for_run(result):
    result = f"I will use the following {result}"
    explanation, code = result.split("Answer:")
    explanation = explanation.strip()
    code = code.strip()

    code_lines = code.split("\n")
    if code_lines[0] in ["```", "```py", "```python"]:
        code_lines = code_lines[1:]
    if code_lines[-1] == "```py":
        code_lines = code_lines[:-1]
    code = "\n".join(code_lines)

    # 返回解释部分和代码部分
    return explanation, code


# 代理类，包含主要的 API 方法
class Agent:
    """
    Base class for all agents which contains the main API methods.
    """
    Args:
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.
    """

    def __init__(self, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        # 调用默认工具设置方法
        _setup_default_tools()

        # 获取代理名
        agent_name = self.__class__.__name__
        # 下载对话模板
        self.chat_prompt_template = download_prompt(chat_prompt_template, agent_name, mode="chat")
        # 下载运行模板
        self.run_prompt_template = download_prompt(run_prompt_template, agent_name, mode="run")
        # 复制默认工具到工具箱
        self._toolbox = HUGGINGFACE_DEFAULT_TOOLS.copy()
        # 设置日志输出函数为打印
        self.log = print
        # 如果有额外工具
        if additional_tools is not None:
            # 如果额外工具是列表或元组
            if isinstance(additional_tools, (list, tuple)):
                # 将其转换为字典，以工具名为键
                additional_tools = {t.name: t for t in additional_tools}
            # 如果额外工具不是字典
            elif not isinstance(additional_tools, dict):
                # 将其转换为字典，以工具名为键
                additional_tools = {additional_tools.name: additional_tools}

            # 找到替换工具处理
            replacements = {name: tool for name, tool in additional_tools.items() if name in HUGGINGFACE_DEFAULT_TOOLS}
            # 添加额外工具到工具箱
            self._toolbox.update(additional_tools)
            # 如果有替换工具
            if len(replacements) > 1:
                # 输出警告
                names = "\n".join([f"- {n}: {t}" for n, t in replacements.items()])
                logger.warning(
                    f"The following tools have been replaced by the ones provided in `additional_tools`:\n{names}."
                )
            # 如果只有一个替换工具
            elif len(replacements) == 1:
                # 输出替换信息
                name = list(replacements.keys())[0]
                logger.warning(f"{name} has been replaced by {replacements[name]} as provided in `additional_tools`.")

        # 准备开始新对话
        self.prepare_for_new_chat()

    @property
    def toolbox(self) -> Dict[str, Tool]:
        """Get all tool currently available to the agent"""
        return self._toolbox
```  
    # 格式化提示信息，根据任务和聊天模式生成提示信息
    def format_prompt(self, task, chat_mode=False):
        # 生成工具描述的字符串
        description = "\n".join([f"- {name}: {tool.description}" for name, tool in self.toolbox.items()])
        
        # 如果是聊天模式
        if chat_mode:
            # 如果聊天记录为空，则使用聊天提示模板并替换工具描述
            if self.chat_history is None:
                prompt = self.chat_prompt_template.replace("<<all_tools>>", description)
            else:
                prompt = self.chat_history
            # 将任务信息添加到聊天提示中
            prompt += CHAT_MESSAGE_PROMPT.replace("<<task>>", task)
        else:
            # 如果不是聊天模式，则使用运行提示模板并替换工具描述和任务
            prompt = self.run_prompt_template.replace("<<all_tools>>", description)
            prompt = prompt.replace("<<prompt>>", task)
        return prompt

    # 设置流处理函数，用于处理在LLM中流输出结果（默认是`print`）
    def set_stream(self, streamer):
        """
        Set the function use to stream results (which is `print` by default).

        Args:
            streamer (`callable`): The function to call when streaming results from the LLM.
        """
        self.log = streamer

    # 在聊天中发送新请求给机器人，在历史记录中使用先前的请求
    def chat(self, task, *, return_code=False, remote=False, **kwargs):
        """
        Sends a new request to the agent in a chat. Will use the previous ones in its history.

        Args:
            task (`str`): The task to perform
            return_code (`bool`, *optional*, defaults to `False`):
                Whether to just return code and not evaluate it.
            remote (`bool`, *optional*, defaults to `False`):
                Whether or not to use remote tools (inference endpoints) instead of local ones.
            kwargs (additional keyword arguments, *optional*):
                Any keyword argument to send to the agent when evaluating the code.

        Example:

        ```py
        from transformers import HfAgent

        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        agent.chat("Draw me a picture of rivers and lakes")

        agent.chat("Transform the picture so that there is a rock in there")
        ```
        """
        # 格式化提示信息
        prompt = self.format_prompt(task, chat_mode=True)
        # 生成一个提示信息并获取结果
        result = self.generate_one(prompt, stop=["Human:", "====="])
        self.chat_history = prompt + result.strip() + "\n"
        # 从结果中提取解释和代码
        explanation, code = clean_code_for_chat(result)

        # 打印代理的解释信息
        self.log(f"==Explanation from the agent==\n{explanation}")

        # 如果有生成的代码
        if code is not None:
            # 打印代理生成的代码
            self.log(f"\n\n==Code generated by the agent==\n{code}")
            # 如果不要返回代码
            if not return_code:
                self.log("\n\n==Result==")
                # 解析代码中调用的工具，更新聊天状态参数，返回评估结果
                self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
                self.chat_state.update(kwargs)
                return evaluate(code, self.cached_tools, self.chat_state, chat_mode=True)
            else:
                # 获取生成工具代码和代码
                tool_code = get_tool_creation_code(code, self.toolbox, remote=remote)
                return f"{tool_code}\n{code}"
    # 准备一个新的聊天环境，清空之前的聊天历史
    def prepare_for_new_chat(self):
        # 清空先前调用 [`~Agent.chat`] 的历史记录
        self.chat_history = None
        # 重置聊天状态
        self.chat_state = {}
        # 清空缓存的工具
        self.cached_tools = None
    
    # 发送一个请求给机器人
    def run(self, task, *, return_code=False, remote=False, **kwargs):
        """
        Sends a request to the agent.
    
        Args:
            task (`str`): The task to perform
            return_code (`bool`, *optional*, defaults to `False`):
                Whether to just return code and not evaluate it.
            remote (`bool`, *optional*, defaults to `False`):
                Whether or not to use remote tools (inference endpoints) instead of local ones.
            kwargs (additional keyword arguments, *optional*):
                Any keyword argument to send to the agent when evaluating the code.
    
        Example:
    
        ```py
        from transformers import HfAgent
    
        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        agent.run("Draw me a picture of rivers and lakes")
        ```
        """
        # 格式化提示
        prompt = self.format_prompt(task)
        # 生成一个结果
        result = self.generate_one(prompt, stop=["Task:"])
        # 清洗生成的代码以便执行
        explanation, code = clean_code_for_run(result)
    
        # 打印机器人的解释
        self.log(f"==Explanation from the agent==\n{explanation}")
    
        # 打印机器人生成的代码
        self.log(f"\n\n==Code generated by the agent==\n{code}")
        # 如果不仅仅返回代码，而要执行生成的代码
        if not return_code:
            # 解析代码并获取工具（评估、执行）
            self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
            return evaluate(code, self.cached_tools, state=kwargs.copy())
        else:
            # 获取工具的创建代码
            tool_code = get_tool_creation_code(code, self.toolbox, remote=remote)
            return f"{tool_code}\n{code}"
    
    # 生成一个结果
    def generate_one(self, prompt, stop):
        # 实现一个定制的代理的方法
        raise NotImplementedError
    
    # 生成多个结果
    def generate_many(self, prompts, stop):
        # 如果有一种更快的批量生成方法，则重写这个方法
        return [self.generate_one(prompt, stop) for prompt in prompts]
class OpenAiAgent(Agent):
    """
    Agent that uses the openai API to generate code.

    <Tip warning={true}>

    The openAI models are used in generation mode, so even for the `chat()` API, it's better to use models like
    `"text-davinci-003"` over the chat-GPT variant. Proper support for chat-GPT models will come in a next version.

    </Tip>

    Args:
        model (`str`, *optional*, defaults to `"text-davinci-003"`):
            The name of the OpenAI model to use.
        api_key (`str`, *optional*):
            The API key to use. If unset, will look for the environment variable `"OPENAI_API_KEY"`.
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from transformers import OpenAiAgent

    agent = OpenAiAgent(model="text-davinci-003", api_key=xxx)
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(
        self,
        model="text-davinci-003",
        api_key=None,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    ):
        # 检查是否安装了 openai 库
        if not is_openai_available():
            raise ImportError("Using `OpenAiAgent` requires `openai`: `pip install openai`.")

        # 如果未提供 API key，则尝试从环境变量中获取
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        # 如果仍未获取到 API key，���抛出数值错误
        if api_key is None:
            raise ValueError(
                "You need an openai key to use `OpenAIAgent`. You can get one here: Get one here "
                "https://openai.com/api/`. If you have one, set it in your env with `os.environ['OPENAI_API_KEY'] = "
                "xxx."
            )
        else:
            # 设置 openai 的 API key
            openai.api_key = api_key
        # 设置模型名称
        self.model = model
        # 调用父类的初始化方法，传递参数
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )
    # 生成多个对话或完成文本的方法
    def generate_many(self, prompts, stop):
        # 如果模型中包含"gpt"，则调用_chat_generate方法生成多个结果
        if "gpt" in self.model:
            return [self._chat_generate(prompt, stop) for prompt in prompts]
        # 否则调用_completion_generate方法生成多个结果
        else:
            return self._completion_generate(prompts, stop)

    # 生成一个对话或完成文本的方法
    def generate_one(self, prompt, stop):
        # 如果模型中包含"gpt"，则调用_chat_generate方法生成一个结果
        if "gpt" in self.model:
            return self._chat_generate(prompt, stop)
        # 否则调用_completion_generate方法生成一个结果
        else:
            return self._completion_generate([prompt], stop)[0]

    # 生成对话文本的方法
    def _chat_generate(self, prompt, stop):
        # 调用OpenAI的chat API生成对话文本
        result = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=stop,
        )
        # 返回生成的对话文本
        return result.choices[0].message.content

    # 生成完成文本的方法
    def _completion_generate(self, prompts, stop):
        # 调用OpenAI的Completion API生成完成文本
        result = openai.Completion.create(
            model=self.model,
            prompt=prompts,
            temperature=0,
            stop=stop,
            max_tokens=200,
        )
        # 返回生成的完成文本列表
        return [answer["text"] for answer in result["choices"]]
class AzureOpenAiAgent(Agent):
    """
    Agent that uses Azure OpenAI to generate code. See the [official
    documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/) to learn how to deploy an openAI
    model on Azure

    <Tip warning={true}>

    The openAI models are used in generation mode, so even for the `chat()` API, it's better to use models like
    `"text-davinci-003"` over the chat-GPT variant. Proper support for chat-GPT models will come in a next version.

    </Tip>

    Args:
        deployment_id (`str`):
            The name of the deployed Azure openAI model to use.
        api_key (`str`, *optional*):
            The API key to use. If unset, will look for the environment variable `"AZURE_OPENAI_API_KEY"`.
        resource_name (`str`, *optional*):
            The name of your Azure OpenAI Resource. If unset, will look for the environment variable
            `"AZURE_OPENAI_RESOURCE_NAME"`.
        api_version (`str`, *optional*, default to `"2022-12-01"`):
            The API version to use for this agent.
        is_chat_mode (`bool`, *optional*):
            Whether you are using a completion model or a chat model (see note above, chat models won't be as
            efficient). Will default to `gpt` being in the `deployment_id` or not.
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from transformers import AzureOpenAiAgent

    agent = AzureAiAgent(deployment_id="Davinci-003", api_key=xxx, resource_name=yyy)
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(
        self,
        deployment_id,
        api_key=None,
        resource_name=None,
        api_version="2022-12-01",
        is_chat_model=None,
        chat_prompt_template=None,
        run_prompt_template=None,
        additional_tools=None,
    # 如果 OpenAI 不可用，则抛出 ImportError 异常
    if not is_openai_available():
        raise ImportError("Using `OpenAiAgent` requires `openai`: `pip install openai`.")

    # 设置部署 ID
    self.deployment_id = deployment_id
    # 设置 OpenAI API 类型为 "azure"
    openai.api_type = "azure"
    
    # 如果未提供 API 密钥，则尝试从环境变量中获取
    if api_key is None:
        api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
    # 如果仍未提供 API 密钥，则抛出 ValueError 异常
    if api_key is None:
        raise ValueError(
            "You need an Azure openAI key to use `AzureOpenAIAgent`. If you have one, set it in your env with "
            "`os.environ['AZURE_OPENAI_API_KEY'] = xxx."
        )
    else:
        openai.api_key = api_key
    
    # 如果未提供资源名称，则尝试从环境变量中获取
    if resource_name is None:
        resource_name = os.environ.get("AZURE_OPENAI_RESOURCE_NAME", None)
    # 如果仍未提供资源名称，则抛出 ValueError 异常
    if resource_name is None:
        raise ValueError(
            "You need a resource_name to use `AzureOpenAIAgent`. If you have one, set it in your env with "
            "`os.environ['AZURE_OPENAI_RESOURCE_NAME'] = xxx."
        )
    else:
        openai.api_base = f"https://{resource_name}.openai.azure.com"
    
    # 设置 API 版本
    openai.api_version = api_version

    # 如果 is_chat_model 未指定，则根据 deployment_id 是否包含 "gpt" 来判断
    if is_chat_model is None:
        is_chat_model = "gpt" in deployment_id.lower()
    self.is_chat_model = is_chat_model

    # 调用父类的构造函数，传入聊天提示模板、运行提示模板和额外工具
    super().__init__(
        chat_prompt_template=chat_prompt_template,
        run_prompt_template=run_prompt_template,
        additional_tools=additional_tools,
    )

def generate_many(self, prompts, stop):
    # 如果是聊天模型，则对每个提示生成聊天结果
    if self.is_chat_model:
        return [self._chat_generate(prompt, stop) for prompt in prompts]
    else:
        return self._completion_generate(prompts, stop)

def generate_one(self, prompt, stop):
    # 如果是聊天模型，则生成单个聊天结果
    if self.is_chat_model:
        return self._chat_generate(prompt, stop)
    else:
        return self._completion_generate([prompt], stop)[0]

def _chat_generate(self, prompt, stop):
    # 使用 OpenAI 的 ChatCompletion API 生成聊天结果
    result = openai.ChatCompletion.create(
        engine=self.deployment_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stop=stop,
    )
    return result["choices"][0]["message"]["content"]

def _completion_generate(self, prompts, stop):
    # 使用 OpenAI 的 Completion API 生成结果
    result = openai.Completion.create(
        engine=self.deployment_id,
        prompt=prompts,
        temperature=0,
        stop=stop,
        max_tokens=200,
    )
    return [answer["text"] for answer in result["choices"]]
class HfAgent(Agent):
    """
    Agent that uses an inference endpoint to generate code.

    Args:
        url_endpoint (`str`):
            The name of the url endpoint to use.
        token (`str`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If unset, will use the token generated when
            running `huggingface-cli login` (stored in `~/.huggingface`).
        chat_prompt_template (`str`, *optional`):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional`):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional`):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    from transformers import HfAgent

    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(
        self, url_endpoint, token=None, chat_prompt_template=None, run_prompt_template=None, additional_tools=None
    ):
        # 初始化函数，设置推理端点的 URL
        self.url_endpoint = url_endpoint
        # 如果未提供 token，则使用 `HfFolder().get_token()` 获取的 token
        if token is None:
            self.token = f"Bearer {HfFolder().get_token()}"
        # 如果 token 以 "Bearer" 或 "Basic" 开头，则直接使用该 token
        elif token.startswith("Bearer") or token.startswith("Basic"):
            self.token = token
        # 否则，将 token 添加 "Bearer " 前缀后使用
        else:
            self.token = f"Bearer {token}"
        # 调用父类的初始化函数，传入 chat_prompt_template、run_prompt_template 和 additional_tools 参数
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )
    # 生成一个文本，根据给定的提示和停止序列
    def generate_one(self, prompt, stop):
        # 设置请求头部，包含授权信息
        headers = {"Authorization": self.token}
        # 设置请求体，包含输入提示、参数和停止序列
        inputs = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "return_full_text": False, "stop": stop},
        }

        # 发送 POST 请求到推理 API，获取响应
        response = requests.post(self.url_endpoint, json=inputs, headers=headers)
        # 处理响应状态码为 429 的情况，表示被限流
        if response.status_code == 429:
            logger.info("Getting rate-limited, waiting a tiny bit before trying again.")
            time.sleep(1)
            # 重新调用生成方法
            return self._generate_one(prompt)
        # 处理非 200 状态码的情况，抛出异常
        elif response.status_code != 200:
            raise ValueError(f"Error {response.status_code}: {response.json()}")

        # 解析响应数据，获取生成的文本
        result = response.json()[0]["generated_text"]
        # 检查生成的文本是否以停止序列结尾，如果是则去除停止序列
        for stop_seq in stop:
            if result.endswith(stop_seq):
                return result[: -len(stop_seq)]
        return result
# 定义一个名为 LocalAgent 的类，继承自 Agent 类
class LocalAgent(Agent):
    """
    Agent that uses a local model and tokenizer to generate code.

    Args:
        model ([`PreTrainedModel`]):
            The model to use for the agent.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer to use for the agent.
        chat_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `chat` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `chat_prompt_template.txt` in this repo in this case.
        run_prompt_template (`str`, *optional*):
            Pass along your own prompt if you want to override the default template for the `run` method. Can be the
            actual prompt template or a repo ID (on the Hugging Face Hub). The prompt should be in a file named
            `run_prompt_template.txt` in this repo in this case.
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            Any additional tools to include on top of the default ones. If you pass along a tool with the same name as
            one of the default tools, that default tool will be overridden.

    Example:

    ```py
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LocalAgent

    checkpoint = "bigcode/starcoder"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    agent = LocalAgent(model, tokenizer)
    agent.run("Draw me a picture of rivers and lakes.")
    ```
    """

    # 初始化方法，接受模型、分词器、聊天提示模板、运行提示模板和额外工具作为参数
    def __init__(self, model, tokenizer, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        # 将传入的模型和分词器赋值给实例变量
        self.model = model
        self.tokenizer = tokenizer
        # 调用父类 Agent 的初始化方法，传入聊天提示模板、运行提示模板和额外工具
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    # 类方法
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Convenience method to build a `LocalAgent` from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The name of a repo on the Hub or a local path to a folder containing both model and tokenizer.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments passed along to [`~PreTrainedModel.from_pretrained`].

        Example:

        ```py
        import torch
        from transformers import LocalAgent

        agent = LocalAgent.from_pretrained("bigcode/starcoder", device_map="auto", torch_dtype=torch.bfloat16)
        agent.run("Draw me a picture of rivers and lakes.")
        ```
        """
        # 从预训练模型名称或路径构建一个`LocalAgent`的便捷方法
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(model, tokenizer)

    @property
    def _model_device(self):
        # 检查模型是否有`hf_device_map`属性，返回设备映射的第一个值
        if hasattr(self.model, "hf_device_map"):
            return list(self.model.hf_device_map.values())[0]
        # 遍历模型参数，返回参数所在的设备
        for param in self.model.parameters():
            return param.device

    def generate_one(self, prompt, stop):
        # 对提示进行编码，返回PyTorch张量，并将其移动到模型设备上
        encoded_inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_device)
        src_len = encoded_inputs["input_ids"].shape[1]
        # 定义停止生成的条件列表
        stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop, self.tokenizer)])
        # 生成输出序列
        outputs = self.model.generate(
            encoded_inputs["input_ids"], max_new_tokens=200, stopping_criteria=stopping_criteria
        )

        # 解码输出序列，去除源长度部分
        result = self.tokenizer.decode(outputs[0].tolist()[src_len:])
        # 推理API返回停止序列
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        return result
# 定义一个继承自 StoppingCriteria 的类 StopSequenceCriteria，用于在遇到特定序列时停止生成
class StopSequenceCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever a sequence of tokens is encountered.

    Args:
        stop_sequences (`str` or `List[str]`):
            The sequence (or list of sequences) on which to stop execution.
        tokenizer:
            The tokenizer used to decode the model outputs.
    """

    # 初始化方法，接受停止序列和分词器作为参数
    def __init__(self, stop_sequences, tokenizer):
        # 如果停止序列是字符串，则转换为列表
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        # 设置停止序列和分词器属性
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer

    # 调用方法，判断是否应该停止生成
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # 解码模型输出的 input_ids，转换为字符串
        decoded_output = self.tokenizer.decode(input_ids.tolist()[0])
        # 判断是否有任何停止序列在解码后的输出中出现
        return any(decoded_output.endswith(stop_sequence) for stop_sequence in self.stop_sequences)
```