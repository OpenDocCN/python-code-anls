# `.\DB-GPT-src\dbgpt\app\chat_adapter.py`

```py
"""
This code file will be deprecated in the future. 
We have integrated fastchat. For details, see: dbgpt/model/model_adapter.py
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import cache  # 导入 functools 模块中的 cache 装饰器
from typing import Dict, List, Tuple  # 导入类型提示所需的类型定义

from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType  # 导入消息相关类
from dbgpt.model.llm.conversation import Conversation, get_conv_template  # 导入对话相关类和获取对话模板函数


class BaseChatAdpter:
    """The Base class for chat with llm models. it will match the model,
    and fetch output from model"""

    def match(self, model_path: str):
        """Check if the provided model path matches with this adapter."""
        return False  # 默认返回 False，子类需要实现具体匹配逻辑

    def get_generate_stream_func(self, model_path: str):
        """Return the generate stream handler func."""
        from dbgpt.model.llm.inference import generate_stream  # 从推断模块导入生成流函数

        return generate_stream  # 返回生成流函数的引用

    def get_conv_template(self, model_path: str) -> Conversation:
        """Retrieve conversation template for the given model path."""
        return None  # 默认返回空，子类需要实现具体的获取对话模板逻辑

    def model_adaptation(
        self, params: Dict, model_path: str, prompt_template: str = None
    ):
        """
        Adapt the model using provided parameters and model path.
        
        This method is intended to handle model adaptation based on given parameters
        and model path, optionally using a prompt template.
        """
    ) -> Tuple[Dict, Dict]:
        """定义函数签名，返回类型为元组，包含两个字典"""

        # 获取模型路径下的会话模板
        conv = self.get_conv_template(model_path)

        # 从参数中获取消息列表
        messages = params.get("messages")

        # 设置模型上下文，用于与 dbgpt 服务器交互
        model_context = {"prompt_echo_len_char": -1}

        if messages:
            # 将消息列表转换为 ModelMessage 类型的对象列表
            messages = [
                m if isinstance(m, ModelMessage) else ModelMessage(**m)
                for m in messages
            ]
            params["messages"] = messages

        if prompt_template:
            # 如果有指定的提示模板，则使用该模板
            print(f"Use prompt template {prompt_template} from config")
            conv = get_conv_template(prompt_template)

        if not conv or not messages:
            # 如果会话模板或消息列表为空，则无需继续处理，直接返回
            print(
                f"No conv from model_path {model_path} or no messages in params, {self}"
            )
            return params, model_context

        # 复制会话模板，以免修改原始模板
        conv = conv.copy()

        # 存储系统消息的列表
        system_messages = []

        # 遍历消息列表，根据消息类型进行处理
        for message in messages:
            role, content = None, None

            if isinstance(message, ModelMessage):
                role = message.role
                content = message.content
            elif isinstance(message, dict):
                role = message["role"]
                content = message["content"]
            else:
                raise ValueError(f"Invalid message type: {message}")

            # 根据消息的角色类型进行分类处理
            if role == ModelMessageRoleType.SYSTEM:
                # 支持多个系统消息，将内容添加到系统消息列表中
                system_messages.append(content)
            elif role == ModelMessageRoleType.HUMAN:
                # 将人类角色的内容添加到会话模板的人类角色中
                conv.append_message(conv.roles[0], content)
            elif role == ModelMessageRoleType.AI:
                # 将 AI 角色的内容添加到会话模板的 AI 角色中
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")

        if system_messages:
            # 如果存在系统消息，则将所有系统消息内容合并，并更新到会话模板中
            conv.update_system_message("".join(system_messages))

        # 为助手添加一个空白消息
        conv.append_message(conv.roles[1], None)

        # 获取更新后的提示内容
        new_prompt = conv.get_prompt()

        # 计算更新后的提示内容的长度（去除特定标记）
        prompt_echo_len_char = len(new_prompt.replace("</s>", "").replace("<s>", ""))

        # 更新模型上下文中的提示长度信息
        model_context["prompt_echo_len_char"] = prompt_echo_len_char
        model_context["echo"] = params.get("echo", True)

        # 将更新后的提示内容添加到参数中
        params["prompt"] = new_prompt

        # 覆盖模型参数中的停止标记
        params["stop"] = conv.stop_str

        # 返回更新后的参数和模型上下文信息
        return params, model_context
# 定义一个空列表，用于存储聊天适配器对象
llm_model_chat_adapters: List[BaseChatAdpter] = []

# 注册一个聊天适配器
def register_llm_model_chat_adapter(cls):
    """Register a chat adapter"""
    llm_model_chat_adapters.append(cls())

# 使用缓存装饰器，获取指定模型的聊天生成函数
@cache
def get_llm_chat_adapter(model_name: str, model_path: str) -> BaseChatAdpter:
    """Get a chat generate func for a model"""
    # 遍历已注册的聊天适配器，根据模型名称匹配适配器
    for adapter in llm_model_chat_adapters:
        if adapter.match(model_name):
            print(f"Get model chat adapter with model name {model_name}, {adapter}")
            return adapter
    # 若模型名称匹配失败，则根据模型路径匹配适配器
    for adapter in llm_model_chat_adapters:
        if adapter.match(model_path):
            print(f"Get model chat adapter with model path {model_path}, {adapter}")
            return adapter
    # 若都匹配失败，则抛出值错误异常
    raise ValueError(
        f"Invalid model for chat adapter with model name {model_name} and model path {model_path}"
    )

# 定义维库纳聊天适配器类，继承自基础聊天适配器类
class VicunaChatAdapter(BaseChatAdpter):
    """Model chat Adapter for vicuna"""

    # 判断模型路径是否基于 Llama2
    def _is_llama2_based(self, model_path: str):
        # see https://huggingface.co/lmsys/vicuna-13b-v1.5
        return "v1.5" in model_path.lower()

    # 匹配模型路径是否包含维库纳
    def match(self, model_path: str):
        return "vicuna" in model_path.lower()

    # 获取对话模板
    def get_conv_template(self, model_path: str) -> Conversation:
        if self._is_llama2_based(model_path):
            return get_conv_template("vicuna_v1.1")
        return None

    # 获取生成流函数
    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.vicuna_base_llm import generate_stream

        if self._is_llama2_based(model_path):
            return super().get_generate_stream_func(model_path)
        return generate_stream

# 定义 ChatGLM 聊天适配器类，继承自基础聊天适配器类
class ChatGLMChatAdapter(BaseChatAdpter):
    """Model chat Adapter for ChatGLM"""

    # 匹配模型路径是否包含 ChatGLM
    def match(self, model_path: str):
        return "chatglm" in model_path

    # 获取生成流函数
    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.chatglm_llm import chatglm_generate_stream

        return chatglm_generate_stream

# 定义 Guanaco 聊天适配器类，继承自基础聊天适配器类
class GuanacoChatAdapter(BaseChatAdpter):
    """Model chat adapter for Guanaco"""

    # 匹配模型路径是否包含 Guanaco
    def match(self, model_path: str):
        return "guanaco" in model_path

    # 获取生成流函数
    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.guanaco_llm import guanaco_generate_stream

        return guanaco_generate_stream

# 定义 Falcon 聊天适配器类，继承自基础聊天适配器类
class FalconChatAdapter(BaseChatAdpter):
    """Model chat adapter for Guanaco"""

    # 匹配模型路径是否包含 Falcon
    def match(self, model_path: str):
        return "falcon" in model_path

    # 获取生成流函数
    def get_generate_stream_func(self, model_path: str):
        from dbgpt.model.llm_out.falcon_llm import falcon_generate_output

        return falcon_generate_output

# 定义 Gorilla 聊天适配器类，继承自基础聊天适配器类
class GorillaChatAdapter(BaseChatAdpter:
    # 判断给定的模型路径字符串中是否包含子字符串 "gorilla"
    def match(self, model_path: str):
        return "gorilla" in model_path

    # 根据模型路径动态导入特定模块中的函数，并返回该函数对象
    def get_generate_stream_func(self, model_path: str):
        # 从特定模块中导入 generate_stream 函数
        from dbgpt.model.llm_out.gorilla_llm import generate_stream

        # 返回导入的 generate_stream 函数对象
        return generate_stream
class GPT4AllChatAdapter(BaseChatAdpter):
    # GPT-4 All 模型适配器，继承自 BaseChatAdpter
    def match(self, model_path: str):
        # 判断给定的模型路径是否包含 "gptj-6b"
        return "gptj-6b" in model_path

    def get_generate_stream_func(self, model_path: str):
        # 导入并返回 gpt4all_generate_stream 函数，用于生成数据流
        from dbgpt.model.llm_out.gpt4all_llm import gpt4all_generate_stream
        return gpt4all_generate_stream


class Llama2ChatAdapter(BaseChatAdpter):
    # Llama-2 模型适配器，继承自 BaseChatAdpter
    def match(self, model_path: str):
        # 判断给定的模型路径是否包含 "llama-2"（不区分大小写）
        return "llama-2" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        # 返回使用 "llama-2" 模板的会话对象
        return get_conv_template("llama-2")


class CodeLlamaChatAdapter(BaseChatAdpter):
    # CodeLlama 模型适配器，继承自 BaseChatAdpter
    """The model ChatAdapter for codellama ."""

    def match(self, model_path: str):
        # 判断给定的模型路径是否包含 "codellama"（不区分大小写）
        return "codellama" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        # 返回使用 "codellama" 模板的会话对象
        return get_conv_template("codellama")


class BaichuanChatAdapter(BaseChatAdpter):
    # Baichuan 模型适配器，继承自 BaseChatAdpter
    def match(self, model_path: str):
        # 判断给定的模型路径是否包含 "baichuan"（不区分大小写）
        return "baichuan" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        # 如果模型路径中包含 "chat"，返回使用 "baichuan-chat" 模板的会话对象；否则返回 "zero_shot" 模板的会话对象
        if "chat" in model_path.lower():
            return get_conv_template("baichuan-chat")
        return get_conv_template("zero_shot")


class WizardLMChatAdapter(BaseChatAdpter):
    # WizardLM 模型适配器，继承自 BaseChatAdpter
    def match(self, model_path: str):
        # 判断给定的模型路径是否包含 "wizardlm"（不区分大小写）
        return "wizardlm" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        # 返回使用 "vicuna_v1.1" 模板的会话对象
        return get_conv_template("vicuna_v1.1")


class LlamaCppChatAdapter(BaseChatAdpter):
    # LlamaCpp 模型适配器，继承自 BaseChatAdpter
    def match(self, model_path: str):
        # 导入 LlamaCppAdapater 类并判断给定的模型路径是否是 "llama-cpp"
        from dbgpt.model.adapter.old_adapter import LlamaCppAdapater
        if "llama-cpp" == model_path:
            return True
        # 判断给定的模型路径是否匹配特定格式，并返回匹配结果
        is_match, _ = LlamaCppAdapater._parse_model_path(model_path)
        return is_match

    def get_conv_template(self, model_path: str) -> Conversation:
        # 返回使用 "llama-2" 模板的会话对象
        return get_conv_template("llama-2")

    def get_generate_stream_func(self, model_path: str):
        # 导入并返回 generate_stream 函数，用于生成数据流
        from dbgpt.model.llm_out.llama_cpp_llm import generate_stream
        return generate_stream


class InternLMChatAdapter(BaseChatAdpter):
    # InternLM 模型适配器，继承自 BaseChatAdpter
    """The model adapter for internlm/internlm-chat-7b"""

    def match(self, model_path: str):
        # 判断给定的模型路径是否包含 "internlm"（不区分大小写）
        return "internlm" in model_path.lower()

    def get_conv_template(self, model_path: str) -> Conversation:
        # 返回使用 "internlm-chat" 模板的会话对象
        return get_conv_template("internlm-chat")


# 向注册系统注册不同的模型适配器
register_llm_model_chat_adapter(VicunaChatAdapter)
register_llm_model_chat_adapter(ChatGLMChatAdapter)
register_llm_model_chat_adapter(GuanacoChatAdapter)
register_llm_model_chat_adapter(FalconChatAdapter)
register_llm_model_chat_adapter(GorillaChatAdapter)
register_llm_model_chat_adapter(GPT4AllChatAdapter)
register_llm_model_chat_adapter(Llama2ChatAdapter)
register_llm_model_chat_adapter(CodeLlamaChatAdapter)
register_llm_model_chat_adapter(BaichuanChatAdapter)
register_llm_model_chat_adapter(WizardLMChatAdapter)
register_llm_model_chat_adapter(LlamaCppChatAdapter)
register_llm_model_chat_adapter(InternLMChatAdapter)

# 用于测试和开发的代理模型，目前对我们来说成本较低。
# 注册一个基于 LLM 的聊天适配器，使用 BaseChatAdpter 作为基类
register_llm_model_chat_adapter(BaseChatAdpter)
```