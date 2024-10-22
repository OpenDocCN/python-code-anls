# `.\chatglm4-finetune\composite_demo\src\conversation.py`

```py
# 导入 JSON 模块，用于处理 JSON 数据
import json
# 导入正则表达式模块，用于字符串匹配
import re
# 从 dataclasses 模块导入 dataclass 装饰器，用于简化数据类的定义
from dataclasses import dataclass
# 从 datetime 模块导入 datetime 类，用于处理日期和时间
from datetime import datetime
# 从 enum 模块导入 Enum 和 auto，分别用于定义枚举和自动赋值
from enum import Enum, auto

# 导入 Streamlit 库，用于构建网页应用
import streamlit as st
# 从 Streamlit 的 delta_generator 模块导入 DeltaGenerator，用于动态内容生成
from streamlit.delta_generator import DeltaGenerator

# 从 PIL.Image 导入 Image 类，用于处理图像
from PIL.Image import Image

# 从 tools.browser 导入 Quote 和 quotes，用于处理引用和引用列表
from tools.browser import Quote, quotes

# 定义一个正则表达式，用于匹配特定格式的引用
QUOTE_REGEX = re.compile(r"【(\d+)†(.+?)】")

# 定义自我介绍提示，说明该助手的身份和任务
SELFCOG_PROMPT = "你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。"
# 定义日期提示的格式
DATE_PROMPT = "当前日期: %Y-%m-%d"
# 定义工具系统提示，包含不同工具的说明
TOOL_SYSTEM_PROMPTS = {
    "python": "当你向 `python` 发送包含 Python 代码的消息时，该代码将会在一个有状态的 Jupyter notebook 环境中执行。\n`python` 返回代码执行的输出，或在执行 60 秒后返回超时。\n`/mnt/data` 将会持久化存储你的文件。在此会话中，`python` 无法访问互联网。不要使用 `python` 进行任何网络请求或者在线 API 调用，这些在线内容的访问将不会成功。",
    "simple_browser": "你可以使用 `simple_browser` 工具。该工具支持以下函数：\n`search(query: str, recency_days: int)`：使用搜索引擎进行查询并显示结果，可以使用 `recency_days` 参数控制搜索内容的时效性。\n`mclick(ids: list[int])`：获取一系列指定 id 的页面内容。每次调用时，须选择3-10个页面。选择多个角度的页面，同时尽可能选择可信任的信息来源。考虑到部分页面是无法加载的，你也可以多打开一些可能有用的页面而不用担心内容过多。\n`open_url(url: str)`：打开指定的 URL。\n\n使用 `【{引用 id}†{引用文本}】` 来引用内容。\n\n操作步骤：1. 使用 `search` 来获得信息列表; 2. 使用 `mclick` 来获取指定 ID 页面的内容; 3. 根据获得的内容进行回复。在回复中应当引用信息来源。\n 如果用户提供了 URL，也可以用 `open_url` 直接打开页面。\n如果初次搜索结果没有找到合适的信息，也可以再次使用 `search` 进行搜索。",
    "cogview": "如果用户的请求中包含了对图像的描述，你可以使用 `cogview` 来生成图像并展示给用户。你需要向 `cogview` 发送图像描述，规则：\n- 发送给 `cogview` 的消息必须使用英语。用户的中文描述必须完全翻译为英语。\n- 应当尽可能详细地描述图像生成的需求，需求描述约 100 英文单词。\n- 保持用户原始描述的意图。不要虚构内容或者没见过的人物。\n- 如无特殊说明，所在地为中国，持有中国立场并遵循中国社会主义价值观。",
}

# 定义文件模板字符串，用于生成文件内容
FILE_TEMPLATE = "[File Name]\n{file_name}\n[File Content]\n{file_content}"


# 定义构建系统提示的函数，接收可用工具和函数列表
def build_system_prompt(
    enabled_tools: list[str],  # 可用工具列表
    functions: list[dict],      # 函数列表
):
    # 初始化提示内容为自我介绍
    value = SELFCOG_PROMPT
    # 将当前日期添加到提示内容中
    value += "\n\n" + datetime.now().strftime(DATE_PROMPT)
    # 如果有可用工具或函数，则添加提示标记
    if enabled_tools or functions:
        value += "\n\n# 可用工具"
    # 初始化内容列表
    contents = []
    # 遍历每个可用工具，添加其描述
    for tool in enabled_tools:
        contents.append(f"\n\n## {tool}\n\n{TOOL_SYSTEM_PROMPTS[tool]}")
    # 遍历每个函数，添加其描述和调用说明
    for function in functions:
        content = f"\n\n## {function['name']}\n\n{json.dumps(function, ensure_ascii=False, indent=4)}"
        content += "\n在调用上述函数时，请使用 Json 格式表示调用的参数。"
        contents.append(content)
    # 将所有内容合并到提示中
    value += "".join(contents)
    # 返回构建的系统提示
    return value


# 定义将响应转换为字符串的函数，支持字符串或字典类型
def response_to_str(response: str | dict[str, str]) -> str:
    """
    将响应转换为字符串。
    """
    # 如果响应是字典类型，则提取名称和内容
    if isinstance(response, dict):
        return response.get("name", "") + response.get("content", "")
    # 如果响应是字符串类型，直接返回
    return response


# 定义角色的枚举类，包含不同角色的定义
class Role(Enum):
    SYSTEM = auto()      # 系统角色
    USER = auto()        # 用户角色
    ASSISTANT = auto()   # 助手角色
    TOOL = auto()        # 工具角色
    OBSERVATION = auto()  # 观察角色

    # 定义角色转换为字符串的方法
    def __str__(self):
        match self:
            case Role.SYSTEM:  # 如果是系统角色，返回对应字符串
                return "<|system|>"
            case Role.USER:    # 如果是用户角色，返回对应字符串
                return "<|user|>"
            case Role.ASSISTANT | Role.TOOL:  # 如果是助手或工具角色，返回对应字符串
                return "<|assistant|>"
            case Role.OBSERVATION:  # 如果是观察角色，返回对应字符串
                return "<|observation|>"

    # 获取给定角色的消息块
    # 定义获取消息的方法
    def get_message(self):
        # 由于 streamlit 的重跑行为，比较值而不是比较对象
        # 因为会话状态中的枚举对象与此处的枚举情况不同
        match self.value:
            # 如果值是系统角色，则不返回任何内容
            case Role.SYSTEM.value:
                return
            # 如果值是用户角色，则返回用户聊天消息
            case Role.USER.value:
                return st.chat_message(name="user", avatar="user")
            # 如果值是助手角色，则返回助手聊天消息
            case Role.ASSISTANT.value:
                return st.chat_message(name="assistant", avatar="assistant")
            # 如果值是工具角色，则返回工具聊天消息
            case Role.TOOL.value:
                return st.chat_message(name="tool", avatar="assistant")
            # 如果值是观察角色，则返回观察聊天消息
            case Role.OBSERVATION.value:
                return st.chat_message(name="observation", avatar="assistant")
            # 如果角色不匹配任何已知情况，则显示错误信息
            case _:
                st.error(f"Unexpected role: {self}")
# 定义一个数据类，用于表示对话内容
@dataclass
class Conversation:
    # 对话的角色（如用户、助手等）
    role: Role
    # 对话的内容，可以是字符串或字典
    content: str | dict
    # 处理过的内容，默认为 None
    saved_content: str | None = None
    # 附加的元数据，默认为 None
    metadata: str | None = None
    # 附带的图像，默认为 None
    image: str | Image | None = None

    # 返回对话对象的字符串表示
    def __str__(self) -> str:
        # 如果有元数据则使用它，否则为空字符串
        metadata_str = self.metadata if self.metadata else ""
        # 格式化并返回角色和内容
        return f"{self.role}{metadata_str}\n{self.content}"

    # 返回人类可读的格式
    def get_text(self) -> str:
        # 使用保存的内容或原始内容
        text = self.saved_content or self.content
        # 根据角色类型决定文本格式
        match self.role.value:
            case Role.TOOL.value:
                # 格式化工具调用的信息
                text = f"Calling tool `{self.metadata}`:\n\n```py\n{text}\n```py"
            case Role.OBSERVATION.value:
                # 格式化观察结果的信息
                text = f"```py\n{text}\n```py"
        # 返回处理后的文本
        return text

    # 以 markdown 块的形式展示内容
    def show(self, placeholder: DeltaGenerator | None = None) -> str:
        # 使用占位符消息或角色消息
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()

        # 如果有图像，则添加图像
        if self.image:
            message.image(self.image, width=512)

        # 如果角色为观察，则格式化消息
        if self.role == Role.OBSERVATION:
            metadata_str = f"from {self.metadata}" if self.metadata else ""
            message = message.expander(f"Observation {metadata_str}")

        # 获取文本内容
        text = self.get_text()
        # 根据角色决定展示的文本内容
        if self.role != Role.USER:
            show_text = text
        else:
            # 分割文本以处理上传的文件内容
            splitted = text.split('files uploaded.\n')
            if len(splitted) == 1:
                show_text = text
            else:
                # 显示文档内容的扩展器
                doc = splitted[0]
                show_text = splitted[-1]
                expander = message.expander(f'File Content')
                expander.markdown(doc)
        # 使用 markdown 格式展示最终文本
        message.markdown(show_text)


# 后处理文本内容的函数
def postprocess_text(text: str, replace_quote: bool) -> str:
    # 替换小括号为美元符号
    text = text.replace("\(", "$")
    # 替换小括号为美元符号
    text = text.replace("\)", "$")
    # 替换中括号为双美元符号
    text = text.replace("\[", "$$")
    # 替换中括号为双美元符号
    text = text.replace("\]", "$$")
    # 移除特定标签
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|observation|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    text = text.replace("<|endoftext|>", "")

    # 如果需要替换引用
    if replace_quote:
        # 遍历找到的引用
        for match in QUOTE_REGEX.finditer(text):
            quote_id = match.group(1)
            # 获取引用内容，如果未找到则使用默认信息
            quote = quotes.get(quote_id, Quote("未找到引用内容", ""))
            # 替换引用文本
            text = text.replace(
                match.group(0), f" (来源：[{quote.title}]({quote.url})) "
            )

    # 返回处理后的文本，去除前后空白
    return text.strip()
```