# `.\tools\agents.py`

```py
# 导入 Python 标准库和第三方模块
import importlib.util  # 导入模块的辅助函数
import json  # 导入 JSON 解析库
import os  # 提供与操作系统交互的功能
import time  # 提供时间相关的功能
from dataclasses import dataclass  # 提供创建数据类的装饰器
from typing import Dict  # 引入类型提示

import requests  # 提供方便的 HTTP 请求功能
from huggingface_hub import HfFolder, hf_hub_download, list_spaces  # 引入与 Hugging Face Hub 相关的功能

# 导入本地定义的模块和函数
from ..models.auto import AutoTokenizer  # 自动加载适合任务的 tokenizer
from ..utils import is_offline_mode, is_openai_available, is_torch_available, logging  # 导入实用工具函数和日志记录器
from .base import TASK_MAPPING, TOOL_CONFIG_FILE, Tool, load_tool, supports_remote  # 导入基础配置和函数定义
from .prompts import CHAT_MESSAGE_PROMPT, download_prompt  # 导入对话提示和下载提示
from .python_interpreter import evaluate  # 导入 Python 解释器相关功能

# 获取日志记录器
logger = logging.get_logger(__name__)

# 如果 OpenAI 可用，导入 openai 模块
if is_openai_available():
    import openai

# 如果 Torch 可用，导入相关模块
if is_torch_available():
    from ..generation import StoppingCriteria, StoppingCriteriaList  # 导入生成停止条件
    from ..models.auto import AutoModelForCausalLM  # 导入适合因果语言建模的自动模型
else:
    StoppingCriteria = object  # 否则定义一个基础对象作为停止条件

# 工具初始化标志
_tools_are_initialized = False

# 基础 Python 工具集合，包括常见内置函数和类型转换
BASE_PYTHON_TOOLS = {
    "print": print,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
}

# 数据类定义，用于表示预设工具的信息
@dataclass
class PreTool:
    task: str  # 工具的任务描述
    description: str  # 工具的描述信息
    repo_id: str  # 工具关联的存储库 ID

# Hugging Face 默认工具的空字典
HUGGINGFACE_DEFAULT_TOOLS = {}

# 从 Hugging Face Hub 导入的默认工具列表
HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB = [
    "image-transformation",
    "text-download",
    "text-to-image",
    "text-to-video",
]

# 获取远程工具的函数定义，从指定组织的存储库中检索工具
def get_remote_tools(organization="huggingface-tools"):
    if is_offline_mode():  # 如果处于离线模式，则提示无法访问远程工具
        logger.info("You are in offline mode, so remote tools are not available.")
        return {}  # 返回空字典表示没有可用的远程工具

    spaces = list_spaces(author=organization)  # 获取指定组织的空间列表
    tools = {}  # 初始化工具字典
    for space_info in spaces:
        repo_id = space_info.id  # 获取存储库 ID
        resolved_config_file = hf_hub_download(repo_id, TOOL_CONFIG_FILE, repo_type="space")  # 下载解析后的配置文件
        with open(resolved_config_file, encoding="utf-8") as reader:
            config = json.load(reader)  # 加载配置文件内容

        task = repo_id.split("/")[-1]  # 提取任务描述
        tools[config["name"]] = PreTool(task=task, description=config["description"], repo_id=repo_id)  # 存储工具信息到字典

    return tools  # 返回工具字典

# 设置默认工具的函数定义
def _setup_default_tools():
    global HUGGINGFACE_DEFAULT_TOOLS  # 声明全局变量
    global _tools_are_initialized  # 声明全局变量

    if _tools_are_initialized:  # 如果工具已初始化，则直接返回，避免重复设置
        return

    main_module = importlib.import_module("transformers")  # 导入 transformers 主模块
    tools_module = main_module.tools  # 获取 tools 子模块

    remote_tools = get_remote_tools()  # 获取远程工具
    # 遍历任务映射中的每个任务名和对应的工具类名
    for task_name, tool_class_name in TASK_MAPPING.items():
        # 从tools_module中获取工具类对象，名称为tool_class_name
        tool_class = getattr(tools_module, tool_class_name)
        # 获取工具类对象的描述信息
        description = tool_class.description
        # 将预处理工具对象加入到HUGGINGFACE_DEFAULT_TOOLS字典中，键为工具类的名称，值为PreTool对象
        HUGGINGFACE_DEFAULT_TOOLS[tool_class.name] = PreTool(task=task_name, description=description, repo_id=None)
    
    # 如果不处于离线模式
    if not is_offline_mode():
        # 遍历需要从Hub获取的默认工具列表中的每个任务名
        for task_name in HUGGINGFACE_DEFAULT_TOOLS_FROM_HUB:
            found = False
            # 遍历远程工具字典中的每个工具名和工具对象
            for tool_name, tool in remote_tools.items():
                # 如果远程工具对象的任务名与当前任务名匹配
                if tool.task == task_name:
                    # 将远程工具对象添加到HUGGINGFACE_DEFAULT_TOOLS字典中，键为工具名
                    HUGGINGFACE_DEFAULT_TOOLS[tool_name] = tool
                    found = True
                    break
    
            # 如果未找到匹配的远程工具，抛出值错误异常
            if not found:
                raise ValueError(f"{task_name} is not implemented on the Hub.")
    
    # 设置工具初始化状态标志为True
    _tools_are_initialized = True
# 解析工具函数，根据给定的代码、工具箱和是否远程访问标志来解析工具
def resolve_tools(code, toolbox, remote=False, cached_tools=None):
    # 如果未提供缓存的工具列表，使用基础 Python 工具的副本作为起点
    if cached_tools is None:
        resolved_tools = BASE_PYTHON_TOOLS.copy()
    else:
        resolved_tools = cached_tools
    # 遍历工具箱中的每个工具项
    for name, tool in toolbox.items():
        # 如果工具名称不在给定代码中，或者已经在解析后的工具列表中，则跳过
        if name not in code or name in resolved_tools:
            continue

        # 如果工具是 Tool 类的实例，直接加入解析后的工具列表
        if isinstance(tool, Tool):
            resolved_tools[name] = tool
        else:
            # 否则根据工具的任务或仓库 ID 加载工具，并根据需要进行远程访问
            task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
            _remote = remote and supports_remote(task_or_repo_id)
            resolved_tools[name] = load_tool(task_or_repo_id, remote=_remote)

    # 返回最终解析后的工具列表
    return resolved_tools


# 生成工具创建代码，根据给定的代码和工具箱
def get_tool_creation_code(code, toolbox, remote=False):
    # 初始化代码行，导入 load_tool 函数
    code_lines = ["from transformers import load_tool", ""]
    # 遍历工具箱中的每个工具项
    for name, tool in toolbox.items():
        # 如果工具名称不在给定代码中或者工具是 Tool 类的实例，则跳过
        if name not in code or isinstance(tool, Tool):
            continue

        # 根据工具的任务或仓库 ID 构建工具创建代码行
        task_or_repo_id = tool.task if tool.repo_id is None else tool.repo_id
        line = f'{name} = load_tool("{task_or_repo_id}"'
        # 如果需要远程访问，则设置 remote=True
        if remote:
            line += ", remote=True"
        line += ")"
        # 添加构建好的代码行到代码列表中
        code_lines.append(line)

    # 将所有代码行连接成一个字符串并返回
    return "\n".join(code_lines) + "\n"


# 清理代码以便于聊天展示
def clean_code_for_chat(result):
    # 拆分结果为解释部分和代码部分
    lines = result.split("\n")
    idx = 0
    # 寻找代码块的起始位置
    while idx < len(lines) and not lines[idx].lstrip().startswith("```"):
        idx += 1
    explanation = "\n".join(lines[:idx]).strip()
    if idx == len(lines):
        return explanation, None

    idx += 1
    start_idx = idx
    # 继续寻找代码块的结束位置
    while not lines[idx].lstrip().startswith("```"):
        idx += 1
    code = "\n".join(lines[start_idx:idx]).strip()

    # 返回清理后的解释部分和代码部分
    return explanation, code


# 清理代码以便于运行
def clean_code_for_run(result):
    # 添加标记到结果中以便于识别和处理
    result = f"I will use the following {result}"
    # 拆分结果为解释部分和代码部分
    explanation, code = result.split("Answer:")
    explanation = explanation.strip()
    code = code.strip()

    # 分割代码行并去除首尾的代码块标记
    code_lines = code.split("\n")
    if code_lines[0] in ["```", "```", "```"]:
        code_lines = code_lines[1:]
    if code_lines[-1] == "```":
        code_lines = code_lines[:-1]
    code = "\n".join(code_lines)

    # 返回清理后的解释部分和代码部分
    return explanation, code
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

    # 定义一个类，用于处理对话生成模型的配置和工具
    def __init__(self, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        # 设置默认工具集
        _setup_default_tools()

        # 获取当前类的名称作为代理名称
        agent_name = self.__class__.__name__

        # 下载指定的对话生成模板，用于聊天模式
        self.chat_prompt_template = download_prompt(chat_prompt_template, agent_name, mode="chat")

        # 下载指定的对话生成模板，用于运行模式
        self.run_prompt_template = download_prompt(run_prompt_template, agent_name, mode="run")

        # 复制默认的 Hugging Face 工具集到代理的工具箱
        self._toolbox = HUGGINGFACE_DEFAULT_TOOLS.copy()

        # 设置日志功能为打印输出
        self.log = print

        # 如果提供了额外的工具，则更新代理的工具箱
        if additional_tools is not None:
            # 如果 additional_tools 是列表或元组，则将其转换为字典，以工具名称作为键
            if isinstance(additional_tools, (list, tuple)):
                additional_tools = {t.name: t for t in additional_tools}
            # 如果 additional_tools 不是字典，则将其转换为包含单个工具的字典
            elif not isinstance(additional_tools, dict):
                additional_tools = {additional_tools.name: additional_tools}

            # 找出在 additional_tools 中已经存在于默认工具集中的工具，并将其用新工具替换
            replacements = {name: tool for name, tool in additional_tools.items() if name in HUGGINGFACE_DEFAULT_TOOLS}
            # 更新代理的工具箱
            self._toolbox.update(additional_tools)

            # 如果有工具被替换了，则记录警告信息
            if len(replacements) > 1:
                names = "\n".join([f"- {n}: {t}" for n, t in replacements.items()])
                logger.warning(
                    f"The following tools have been replaced by the ones provided in `additional_tools`:\n{names}."
                )
            elif len(replacements) == 1:
                name = list(replacements.keys())[0]
                logger.warning(f"{name} has been replaced by {replacements[name]} as provided in `additional_tools`.")

        # 准备进行新的聊天会话的初始化工作
        self.prepare_for_new_chat()

    @property
    def toolbox(self) -> Dict[str, Tool]:
        """Get all tool currently available to the agent"""
        # 返回当前代理可用的所有工具集合
        return self._toolbox
    def format_prompt(self, task, chat_mode=False):
        # 构建描述工具的字符串，每行包含工具名称和描述
        description = "\n".join([f"- {name}: {tool.description}" for name, tool in self.toolbox.items()])
        
        # 根据聊天模式选择不同的提示模板
        if chat_mode:
            # 如果历史记录为空，使用聊天提示模板并替换工具描述部分
            if self.chat_history is None:
                prompt = self.chat_prompt_template.replace("<<all_tools>>", description)
            else:
                # 否则，使用已有的聊天历史记录
                prompt = self.chat_history
            # 添加当前任务到聊天提示中
            prompt += CHAT_MESSAGE_PROMPT.replace("<<task>>", task)
        else:
            # 使用运行提示模板并替换工具描述和任务部分
            prompt = self.run_prompt_template.replace("<<all_tools>>", description)
            prompt = prompt.replace("<<prompt>>", task)
        
        # 返回生成的提示
        return prompt

    def set_stream(self, streamer):
        """
        Set the function use to stream results (which is `print` by default).

        Args:
            streamer (`callable`): The function to call when streaming results from the LLM.
        """
        # 设置日志输出函数，用于从语言模型生成的结果流式输出
        self.log = streamer

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
        """
        # 根据任务构建格式化后的提示字符串，用于发送给语言模型
        prompt = self.format_prompt(task, chat_mode=True)
        
        # 生成一个对话回复，停止条件为指定字符串
        result = self.generate_one(prompt, stop=["Human:", "====="])
        
        # 更新聊天历史记录，包含之前的提示和生成的结果
        self.chat_history = prompt + result.strip() + "\n"
        
        # 清理生成的代码，获取解释和代码
        explanation, code = clean_code_for_chat(result)
        
        # 输出语言模型生成的解释信息
        self.log(f"==Explanation from the agent==\n{explanation}")
        
        # 如果生成了代码，则输出生成的代码，并根据需求返回或评估代码结果
        if code is not None:
            self.log(f"\n\n==Code generated by the agent==\n{code}")
            if not return_code:
                # 解析工具并返回评估结果
                self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
                self.chat_state.update(kwargs)
                return evaluate(code, self.cached_tools, self.chat_state, chat_mode=True)
            else:
                # 获取生成代码的工具创建代码并返回完整的生成代码
                tool_code = get_tool_creation_code(code, self.toolbox, remote=remote)
                return f"{tool_code}\n{code}"
    def prepare_for_new_chat(self):
        """
        Clears the history of prior calls to [`~Agent.chat`].
        """
        # 清空聊天历史记录
        self.chat_history = None
        # 重置聊天状态
        self.chat_state = {}
        # 清空缓存的工具
        self.cached_tools = None

    def clean_code_for_run(self, result):
        """
        Override this method if you want to change the way the code is
        cleaned for the `run` method.
        """
        # 调用特定函数来清理运行代码
        return clean_code_for_run(result)

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

        ```
        from transformers import HfAgent

        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        agent.run("Draw me a picture of rivers and lakes")
        ```
        """
        # 格式化任务提示信息
        prompt = self.format_prompt(task)
        # 生成一次结果，返回解释和生成的代码
        result = self.generate_one(prompt, stop=["Task:"])
        explanation, code = self.clean_code_for_run(result)

        # 记录代理生成的解释
        self.log(f"==Explanation from the agent==\n{explanation}")

        # 记录代理生成的代码
        self.log(f"\n\n==Code generated by the agent==\n{code}")
        if not return_code:
            # 如果不仅返回代码而不评估，则解析工具并评估生成的代码
            self.log("\n\n==Result==")
            self.cached_tools = resolve_tools(code, self.toolbox, remote=remote, cached_tools=self.cached_tools)
            return evaluate(code, self.cached_tools, state=kwargs.copy())
        else:
            # 获取工具创建代码并返回与生成代码的组合结果
            tool_code = get_tool_creation_code(code, self.toolbox, remote=remote)
            return f"{tool_code}\n{code}"

    def generate_one(self, prompt, stop):
        # This is the method to implement in your custom agent.
        # 这是需要在自定义代理中实现的方法，抛出未实现错误
        raise NotImplementedError

    def generate_many(self, prompts, stop):
        # Override if you have a way to do batch generation faster than one by one
        # 如果有批量生成的更快方式，则重写此方法
        return [self.generate_one(prompt, stop) for prompt in prompts]
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
    
    ```
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
        # 检查是否安装了 openai 库，如果没有则抛出 ImportError 异常
        if not is_openai_available():
            raise ImportError("Using `OpenAiAgent` requires `openai`: `pip install openai`.")
        
        # 如果未提供 API 密钥，则尝试从环境变量 "OPENAI_API_KEY" 中获取
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", None)
        
        # 如果仍未设置 API 密钥，则抛出 ValueError 异常
        if api_key is None:
            raise ValueError(
                "You need an openai key to use `OpenAIAgent`. You can get one here: Get one here "
                "https://openai.com/api/`. If you have one, set it in your env with `os.environ['OPENAI_API_KEY'] = "
                "xxx."
            )
        else:
            # 设置 openai 的 API 密钥
            openai.api_key = api_key
        
        # 初始化父类 Agent，传入可能的自定义聊天和运行模板以及额外工具
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )
    # 根据给定的 prompts 列表和 stop 标志，生成多个聊天或完成文本
    def generate_many(self, prompts, stop):
        # 如果模型名称中包含 "gpt"，则使用 _chat_generate 方法生成每个 prompt 的结果
        if "gpt" in self.model:
            return [self._chat_generate(prompt, stop) for prompt in prompts]
        else:
            # 否则，使用 _completion_generate 方法生成所有 prompts 的结果
            return self._completion_generate(prompts, stop)

    # 根据给定的 prompt 和 stop 标志，生成一个聊天或完成文本
    def generate_one(self, prompt, stop):
        # 如果模型名称中包含 "gpt"，则使用 _chat_generate 方法生成结果
        if "gpt" in self.model:
            return self._chat_generate(prompt, stop)
        else:
            # 否则，使用 _completion_generate 方法生成结果，并返回第一个元素
            return self._completion_generate([prompt], stop)[0]

    # 使用 OpenAI 的聊天 API 生成文本
    def _chat_generate(self, prompt, stop):
        # 调用 OpenAI 的 chat.completions.create 方法，传入模型、消息内容、温度和停止条件
        result = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=stop,
        )
        # 返回生成的文本内容
        return result.choices[0].message.content

    # 使用 OpenAI 的 Completion API 生成文本
    def _completion_generate(self, prompts, stop):
        # 调用 OpenAI 的 Completion.create 方法，传入模型、prompt、温度、停止条件和最大 token 数
        result = openai.Completion.create(
            model=self.model,
            prompt=prompts,
            temperature=0,
            stop=stop,
            max_tokens=200,
        )
        # 返回生成的文本内容列表
        return [answer["text"] for answer in result["choices"]]
    """
    AzureOpenAiAgent 是一个继承自 Agent 的代理类，用于利用 Azure OpenAI 生成代码。参考官方文档来了解如何在 Azure 上部署 OpenAI 模型。

    Args:
        deployment_id (`str`):
            要使用的已部署 Azure OpenAI 模型的名称。
        api_key (`str`, *optional*):
            要使用的 API 密钥。如果未设置，将查找环境变量 `"AZURE_OPENAI_API_KEY"`。
        resource_name (`str`, *optional*):
            Azure OpenAI 资源的名称。如果未设置，将查找环境变量 `"AZURE_OPENAI_RESOURCE_NAME"`。
        api_version (`str`, *optional*, default to `"2022-12-01"`):
            该代理使用的 API 版本。
        is_chat_mode (`bool`, *optional*):
            是否使用聊天模型而非完成模型（参见上述注释，聊天模型的效率较低）。默认根据 `deployment_id` 是否包含 `'gpt'` 来判断。
        chat_prompt_template (`str`, *optional*):
            如果要覆盖 `chat` 方法的默认模板，请传递自定义的提示模板。可以是实际的提示模板或 Hugging Face Hub 上的 repo ID。在这种情况下，提示应该在该 repo 中命名为 `chat_prompt_template.txt`。
        run_prompt_template (`str`, *optional*):
            如果要覆盖 `run` 方法的默认模板，请传递自定义的提示模板。可以是实际的提示模板或 Hugging Face Hub 上的 repo ID。在这种情况下，提示应该在该 repo 中命名为 `run_prompt_template.txt`。
        additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
            除默认工具外要包含的任何附加工具。如果传递与默认工具同名的工具，将覆盖默认工具。

    Example:

    ```
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
    ):
        """
        初始化 AzureOpenAiAgent 实例。

        Args:
            deployment_id (`str`):
                要使用的已部署 Azure OpenAI 模型的名称。
            api_key (`str`, *optional*):
                要使用的 API 密钥。如果未设置，将查找环境变量 `"AZURE_OPENAI_API_KEY"`。
            resource_name (`str`, *optional*):
                Azure OpenAI 资源的名称。如果未设置，将查找环境变量 `"AZURE_OPENAI_RESOURCE_NAME"`。
            api_version (`str`, *optional*, default to `"2022-12-01"`):
                该代理使用的 API 版本。
            is_chat_mode (`bool`, *optional*):
                是否使用聊天模型而非完成模型（参见上述注释，聊天模型的效率较低）。默认根据 `deployment_id` 是否包含 `'gpt'` 来判断。
            chat_prompt_template (`str`, *optional*):
                如果要覆盖 `chat` 方法的默认模板，请传递自定义的提示模板。可以是实际的提示模板或 Hugging Face Hub 上的 repo ID。在这种情况下，提示应该在该 repo 中命名为 `chat_prompt_template.txt`。
            run_prompt_template (`str`, *optional*):
                如果要覆盖 `run` 方法的默认模板，请传递自定义的提示模板。可以是实际的提示模板或 Hugging Face Hub 上的 repo ID。在这种情况下，提示应该在该 repo 中命名为 `run_prompt_template.txt`。
            additional_tools ([`Tool`], list of tools or dictionary with tool values, *optional*):
                除默认工具外要包含的任何附加工具。如果传递与默认工具同名的工具，将覆盖默认工具。
        """
        super().__init__()  # 调用父类 Agent 的构造函数
        self.deployment_id = deployment_id
        self.api_key = api_key if api_key else os.getenv("AZURE_OPENAI_API_KEY")  # 设置 API 密钥，如果未提供则从环境变量获取
        self.resource_name = resource_name if resource_name else os.getenv("AZURE_OPENAI_RESOURCE_NAME")  # 设置 Azure OpenAI 资源名称，如果未提供则从环境变量获取
        self.api_version = api_version  # 设置 API 版本
        self.is_chat_mode = is_chat_mode if is_chat_mode is not None else 'gpt' in deployment_id.lower()  # 设置是否为聊天模式，默认根据 deployment_id 是否包含 'gpt' 来判断
        self.chat_prompt_template = chat_prompt_template  # 设置聊天模式的提示模板
        self.run_prompt_template = run_prompt_template  # 设置运行模式的提示模板
        self.additional_tools = additional_tools  # 设置额外的工具列表或字典
    ):
        # 检查是否安装了 openai 库，如果未安装则抛出 ImportError 异常
        if not is_openai_available():
            raise ImportError("Using `OpenAiAgent` requires `openai`: `pip install openai`.")

        # 设置部署 ID
        self.deployment_id = deployment_id
        # 设置 OpenAI API 类型为 "azure"
        openai.api_type = "azure"
        
        # 如果 API 密钥未提供，则尝试从环境变量中获取 AZURE_OPENAI_API_KEY
        if api_key is None:
            api_key = os.environ.get("AZURE_OPENAI_API_KEY", None)
        # 如果仍然没有 API 密钥，则抛出 ValueError 异常
        if api_key is None:
            raise ValueError(
                "You need an Azure openAI key to use `AzureOpenAIAgent`. If you have one, set it in your env with "
                "`os.environ['AZURE_OPENAI_API_KEY'] = xxx."
            )
        else:
            # 设置 OpenAI API 密钥
            openai.api_key = api_key
        
        # 如果资源名称未提供，则尝试从环境变量中获取 AZURE_OPENAI_RESOURCE_NAME
        if resource_name is None:
            resource_name = os.environ.get("AZURE_OPENAI_RESOURCE_NAME", None)
        # 如果仍然没有资源名称，则抛出 ValueError 异常
        if resource_name is None:
            raise ValueError(
                "You need a resource_name to use `AzureOpenAIAgent`. If you have one, set it in your env with "
                "`os.environ['AZURE_OPENAI_RESOURCE_NAME'] = xxx."
            )
        else:
            # 设置 OpenAI API 基础 URL
            openai.api_base = f"https://{resource_name}.openai.azure.com"
        
        # 设置 OpenAI API 版本
        openai.api_version = api_version

        # 如果 is_chat_model 未提供，则根据 deployment_id 决定是否是聊天模型
        if is_chat_model is None:
            is_chat_model = "gpt" in deployment_id.lower()
        # 设置实例的 is_chat_model 属性
        self.is_chat_model = is_chat_model

        # 调用父类的构造函数，初始化实例
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )

    # 生成多个结果的方法
    def generate_many(self, prompts, stop):
        # 如果是聊天模型，则使用 _chat_generate 方法生成多个结果
        if self.is_chat_model:
            return [self._chat_generate(prompt, stop) for prompt in prompts]
        else:
            # 否则使用 _completion_generate 方法生成多个结果
            return self._completion_generate(prompts, stop)

    # 生成单个结果的方法
    def generate_one(self, prompt, stop):
        # 如果是聊天模型，则使用 _chat_generate 方法生成单个结果
        if self.is_chat_model:
            return self._chat_generate(prompt, stop)
        else:
            # 否则使用 _completion_generate 方法生成单个结果并返回第一个结果
            return self._completion_generate([prompt], stop)[0]

    # 聊天生成方法，使用 OpenAI ChatCompletion API
    def _chat_generate(self, prompt, stop):
        result = openai.ChatCompletion.create(
            engine=self.deployment_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            stop=stop,
        )
        # 返回生成的聊天消息内容
        return result["choices"][0]["message"]["content"]

    # 完整生成方法，使用 OpenAI Completion API
    def _completion_generate(self, prompts, stop):
        result = openai.Completion.create(
            engine=self.deployment_id,
            prompt=prompts,
            temperature=0,
            stop=stop,
            max_tokens=200,
        )
        # 返回生成的每个答案的文本内容列表
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

    ```
    from transformers import HfAgent

    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
    agent.run("Is the following `text` (in Spanish) positive or negative?", text="¡Este es un API muy agradable!")
    ```
    """

    def __init__(
        self, url_endpoint, token=None, chat_prompt_template=None, run_prompt_template=None, additional_tools=None
    ):
        # 设置推理端点的URL
        self.url_endpoint = url_endpoint
        # 根据传入的token参数或者从本地获取的token来设置HTTP授权token
        if token is None:
            self.token = f"Bearer {HfFolder().get_token()}"
        elif token.startswith("Bearer") or token.startswith("Basic"):
            self.token = token
        else:
            self.token = f"Bearer {token}"
        # 调用父类构造函数初始化Agent基类
        super().__init__(
            chat_prompt_template=chat_prompt_template,
            run_prompt_template=run_prompt_template,
            additional_tools=additional_tools,
        )
    # 生成一段文本，使用给定的提示和停止条件
    def generate_one(self, prompt, stop):
        # 设置请求头，包含授权信息
        headers = {"Authorization": self.token}
        # 构造请求体，包含输入提示、生成参数和停止条件
        inputs = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 200, "return_full_text": False, "stop": stop},
        }

        # 发送 POST 请求到推理 API
        response = requests.post(self.url_endpoint, json=inputs, headers=headers)
        # 处理请求返回的状态码
        if response.status_code == 429:
            # 如果返回状态码为 429 表示请求过多，记录日志并等待一秒后重试
            logger.info("Getting rate-limited, waiting a tiny bit before trying again.")
            time.sleep(1)
            return self._generate_one(prompt)
        elif response.status_code != 200:
            # 如果返回状态码不是 200，则抛出异常并附带错误信息
            raise ValueError(f"Error {response.status_code}: {response.json()}")

        # 解析响应内容，获取生成的文本结果
        result = response.json()[0]["generated_text"]
        # 检查生成的文本是否以任一停止序列结尾
        for stop_seq in stop:
            if result.endswith(stop_seq):
                # 如果是，则返回去掉停止序列部分的文本
                return result[: -len(stop_seq)]
        # 如果没有匹配到停止序列，则直接返回生成的文本结果
        return result
    @classmethod
    # 类方法装饰器，用于定义一个类的类方法，即可以不用实例化类就可以调用的方法
    def from_local(cls, model_name_or_path, chat_prompt_template=None, run_prompt_template=None, additional_tools=None):
        # 从本地加载模型和分词器
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # 返回一个新的 LocalAgent 实例，初始化时会传入加载的模型和分词器
        return cls(model, tokenizer, chat_prompt_template, run_prompt_template, additional_tools)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Convenience method to build a `LocalAgent` from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The name of a repo on the Hub or a local path to a folder containing both model and tokenizer.
            kwargs (`Dict[str, Any]`, *optional*):
                Keyword arguments passed along to [`~PreTrainedModel.from_pretrained`].

        Example:

        ```
        import torch
        from transformers import LocalAgent

        agent = LocalAgent.from_pretrained("bigcode/starcoder", device_map="auto", torch_dtype=torch.bfloat16)
        agent.run("Draw me a picture of rivers and lakes.")
        ```
        """
        # 使用预训练模型名称或路径创建 AutoModelForCausalLM 对象
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 使用预训练模型名称或路径创建 AutoTokenizer 对象
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # 返回使用创建的模型和分词器构建的 LocalAgent 对象
        return cls(model, tokenizer)

    @property
    def _model_device(self):
        # 检查模型是否有 hf_device_map 属性，返回第一个设备映射的值
        if hasattr(self.model, "hf_device_map"):
            return list(self.model.hf_device_map.values())[0]
        # 如果模型没有 hf_device_map 属性，则返回第一个参数的设备
        for param in self.model.parameters():
            return param.device

    def generate_one(self, prompt, stop):
        # 使用分词器对提示进行编码，返回张量，并移动到模型所在设备
        encoded_inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_device)
        # 计算输入序列的长度
        src_len = encoded_inputs["input_ids"].shape[1]
        # 创建停止条件列表
        stopping_criteria = StoppingCriteriaList([StopSequenceCriteria(stop, self.tokenizer)])
        # 使用模型生成文本序列，限制最大生成的新令牌数为200，并应用停止条件
        outputs = self.model.generate(
            encoded_inputs["input_ids"], max_new_tokens=200, stopping_criteria=stopping_criteria
        )

        # 解码生成的输出，去除原始输入部分并返回结果
        result = self.tokenizer.decode(outputs[0].tolist()[src_len:])
        # 如果结果以停止序列之一结尾，则去除停止序列并更新结果
        for stop_seq in stop:
            if result.endswith(stop_seq):
                result = result[: -len(stop_seq)]
        # 返回生成的结果文本
        return result
class StopSequenceCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever a sequence of tokens is encountered.

    Args:
        stop_sequences (`str` or `List[str]`):
            The sequence (or list of sequences) on which to stop execution.
        tokenizer:
            The tokenizer used to decode the model outputs.
    """

    def __init__(self, stop_sequences, tokenizer):
        # 如果stop_sequences是单个字符串，则转换为列表
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        # 初始化停止序列和分词器
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # 解码输入的模型输出，转换为字符串
        decoded_output = self.tokenizer.decode(input_ids.tolist()[0])
        # 检查解码后的输出是否以任何停止序列结尾
        return any(decoded_output.endswith(stop_sequence) for stop_sequence in self.stop_sequences)
```