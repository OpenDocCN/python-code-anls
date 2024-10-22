# `.\chatglm4-finetune\composite_demo\src\client.py`

```
# 这是 composite_demo 的客户端部分
"""

# 提供两个客户端，HFClient 和 VLLMClient，用于与模型进行交互
We provide two clients, HFClient and VLLMClient, which are used to interact with the model.

# HFClient 用于与 transformers 后端交互，VLLMClient 用于与 VLLM 模型交互
The HFClient is used to interact with the transformers backend, and the VLLMClient is used to interact with the VLLM model.

"""

# 导入 JSON 模块用于处理 JSON 数据
import json
# 导入 Generator 类型用于类型注解
from collections.abc import Generator
# 导入 deepcopy 函数用于深拷贝对象
from copy import deepcopy
# 导入 Enum 和 auto 用于定义枚举类型
from enum import Enum, auto
# 导入 Protocol 类型用于定义协议
from typing import Protocol

# 导入 Streamlit 库以构建用户界面
import streamlit as st

# 从 conversation 模块导入 Conversation 类和 build_system_prompt 函数
from conversation import Conversation, build_system_prompt
# 从 tools.tool_registry 导入所有工具的注册列表
from tools.tool_registry import ALL_TOOLS

# 定义客户端类型的枚举
class ClientType(Enum):
    # 定义 HF 类型
    HF = auto()
    # 定义 VLLM 类型
    VLLM = auto()
    # 定义 API 类型
    API = auto()

# 定义客户端协议，包含初始化和生成流的方法
class Client(Protocol):
    # 定义初始化方法，接受模型路径
    def __init__(self, model_path: str): ...

    # 定义生成流的方法，接受工具和历史记录
    def generate_stream(
        self,
        tools: list[dict],
        history: list[Conversation],
        **parameters,
    ) -> Generator[tuple[str | dict, list[dict]]]: ...

# 处理输入数据的函数
def process_input(history: list[dict], tools: list[dict], role_name_replace:dict=None) -> list[dict]:
    # 初始化聊天历史列表
    chat_history = []
    # 如果有工具，构建系统提示并添加到聊天历史
    #if len(tools) > 0:
    chat_history.append(
        {"role": "system", "content": build_system_prompt(list(ALL_TOOLS), tools)}
    )

    # 遍历历史对话
    for conversation in history:
        # 清理角色名称
        role = str(conversation.role).removeprefix("<|").removesuffix("|>")
        # 如果提供了角色替换字典，更新角色名称
        if role_name_replace:
            role = role_name_replace.get(role, role)
        # 构建对话项
        item = {
            "role": role,
            "content": conversation.content,
        }
        # 如果有元数据，添加到对话项
        if conversation.metadata:
            item["metadata"] = conversation.metadata
        # 仅对用户角色添加图像
        if role == "user" and conversation.image:
            item["image"] = conversation.image
        # 将对话项添加到聊天历史
        chat_history.append(item)

    # 返回聊天历史
    return chat_history

# 处理响应数据的函数
def process_response(output, history):
    # 初始化内容字符串
    content = ""
    # 深拷贝历史记录以避免修改原始数据
    history = deepcopy(history)
    # 分割输出，处理每个助手响应
    for response in output.split("<|assistant|>"):
        # 如果响应中有换行符
        if "\n" in response:
            # 分割元数据和内容
            metadata, content = response.split("\n", maxsplit=1)
        else:
            # 如果没有换行，元数据为空，内容为响应
            metadata, content = "", response
        # 如果元数据为空，则处理内容
        if not metadata.strip():
            content = content.strip()
            # 将助手的响应添加到历史记录
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            # 替换特定文本
            content = content.replace("[[训练时间]]", "2023年")
        else:
            # 否则，添加元数据和内容到历史记录
            history.append({"role": "assistant", "metadata": metadata, "content": content})
            # 如果历史记录的第一项是系统角色，并且包含工具
            if history[0]["role"] == "system" and "tools" in history[0]:
                # 解析内容为参数
                parameters = json.loads(content)
                content = {"name": metadata.strip(), "parameters": parameters}
            else:
                # 否则，将内容结构化
                content = {"name": metadata.strip(), "content": content}
    # 返回处理后的内容和历史记录
    return content, history

# 缓存资源以提高性能，限制缓存条目数
@st.cache_resource(max_entries=1, show_spinner="Loading model...")
def get_client(model_path, typ: ClientType) -> Client:
    # 根据传入的客户端类型决定使用哪个客户端
        match typ:
            # 匹配到 HF 类型时，导入 HFClient
            case ClientType.HF:
                from clients.hf import HFClient
    
                # 返回 HFClient 实例，传入模型路径
                return HFClient(model_path)
            # 匹配到 VLLM 类型时，尝试导入 VLLMClient
            case ClientType.VLLM:
                try:
                    from clients.vllm import VLLMClient
                # 捕获导入错误，并添加提示信息
                except ImportError as e:
                    e.msg += "; did you forget to install vLLM?"
                    raise
                # 返回 VLLMClient 实例，传入模型路径
                return VLLMClient(model_path)
            # 匹配到 API 类型时，导入 APIClient
            case ClientType.API:
                from clients.openai import APIClient
                # 返回 APIClient 实例，传入模型路径
                return APIClient(model_path)
    
        # 如果没有匹配到支持的客户端类型，抛出未实现错误
        raise NotImplementedError(f"Client type {typ} is not supported.")
```