# `.\DB-GPT-src\dbgpt\model\adapter\base.py`

```py
import logging  # 导入日志模块，用于记录程序运行信息
from abc import ABC, abstractmethod  # 导入抽象基类（ABC）和抽象方法装饰器（abstractmethod）
from typing import Any, Callable, Dict, List, Optional, Tuple, Type  # 导入类型提示相关的模块

from dbgpt.core.interface.message import ModelMessage, ModelMessageRoleType  # 导入消息模块中的模型消息和消息角色类型
from dbgpt.model.adapter.template import (  # 导入模型适配器模板相关的模块
    ConversationAdapter,
    ConversationAdapterFactory,
    get_conv_template,
)
from dbgpt.model.base import ModelType  # 导入模型类型枚举
from dbgpt.model.parameter import (  # 导入模型参数相关的模块
    BaseModelParameters,
    LlamaCppModelParameters,
    ModelParameters,
    ProxyModelParameters,
)

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器

class LLMModelAdapter(ABC):
    """New Adapter for DB-GPT LLM models"""

    model_name: Optional[str] = None  # 模型名称，默认为 None
    model_path: Optional[str] = None  # 模型路径，默认为 None
    conv_factory: Optional[ConversationAdapterFactory] = None  # 会话适配器工厂，用于创建会话适配器的实例，默认为 None
    # TODO: more flexible quantization config
    support_4bit: bool = False  # 是否支持4位量化，默认为 False
    support_8bit: bool = False  # 是否支持8位量化，默认为 False
    support_system_message: bool = True  # 是否支持系统消息，默认为 True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model_name={self.model_name} model_path={self.model_path}>"

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def new_adapter(self, **kwargs) -> "LLMModelAdapter":
        """Create a new adapter instance

        Args:
            **kwargs: The parameters of the new adapter instance

        Returns:
            LLMModelAdapter: The new adapter instance
        """
        # 抽象方法：创建一个新的适配器实例，需要子类实现具体逻辑

    def use_fast_tokenizer(self) -> bool:
        """Whether use a [fast Rust-based tokenizer](https://huggingface.co/docs/tokenizers/index) if it is supported
        for a given model.
        """
        return False  # 默认情况下不使用快速的基于Rust的分词器

    def model_type(self) -> str:
        return ModelType.HF  # 返回模型类型为 HF

    def model_param_class(self, model_type: str = None) -> Type[BaseModelParameters]:
        """Get the startup parameters instance of the model

        Args:
            model_type (str, optional): The type of model. Defaults to None.

        Returns:
            Type[BaseModelParameters]: The startup parameters instance of the model
        """
        model_type = model_type if model_type else self.model_type()  # 如果未指定模型类型，则使用当前实例的模型类型
        if model_type == ModelType.LLAMA_CPP:
            return LlamaCppModelParameters  # 如果模型类型是 LLAMA_CPP，返回 LlamaCppModelParameters 类型
        elif model_type == ModelType.PROXY:
            return ProxyModelParameters  # 如果模型类型是 PROXY，返回 ProxyModelParameters 类型
        return ModelParameters  # 默认情况下返回 ModelParameters 类型

    def match(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> bool:
        """Whether the model adapter can load the given model

        Args:
            model_type (str): The type of model
            model_name (Optional[str], optional): The name of model. Defaults to None.
            model_path (Optional[str], optional): The path of model. Defaults to None.
        """
        return False  # 默认情况下，模型适配器不支持加载任何模型
    def support_quantization_4bit(self) -> bool:
        """Whether the model adapter can load 4bit model

        If it is True, we will load the 4bit model with :meth:`~LLMModelAdapter.load`

        Returns:
            bool: Whether the model adapter can load 4bit model, default is False
        """
        return self.support_4bit



    def support_quantization_8bit(self) -> bool:
        """Whether the model adapter can load 8bit model

        If it is True, we will load the 8bit model with :meth:`~LLMModelAdapter.load`

        Returns:
            bool: Whether the model adapter can load 8bit model, default is False
        """
        return self.support_8bit



    def load(self, model_path: str, from_pretrained_kwargs: dict):
        """Load model and tokenizer

        This method is meant to be implemented by subclasses to load a model and tokenizer
        from the given `model_path` and `from_pretrained_kwargs`.

        Args:
            model_path (str): The path to the model.
            from_pretrained_kwargs (dict): Additional keyword arguments for loading.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError



    def parse_max_length(self, model, tokenizer) -> Optional[int]:
        """Parse the max_length of the model.

        Args:
            model: The model object.
            tokenizer: The tokenizer object.

        Returns:
            Optional[int]: The max_length of the model, or None if not found.
        """
        if not (tokenizer or model):
            return None
        try:
            model_max_length = None
            if tokenizer and hasattr(tokenizer, "model_max_length"):
                model_max_length = tokenizer.model_max_length
            if model_max_length and model_max_length < 100000000:
                # Can't be too large
                return model_max_length
            if model and hasattr(model, "config"):
                model_config = model.config
                if hasattr(model_config, "max_sequence_length"):
                    return model_config.max_sequence_length
                if hasattr(model_config, "max_position_embeddings"):
                    return model_config.max_position_embeddings
            return None
        except Exception:
            return None



    def load_from_params(self, params):
        """Load the model and tokenizer according to the given parameters

        This method is meant to be implemented by subclasses to load a model and tokenizer
        based on the parameters provided.

        Args:
            params: Parameters for loading the model and tokenizer.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError



    def support_async(self) -> bool:
        """Whether the loaded model supports asynchronous calls

        Returns:
            bool: False, indicating asynchronous calls are not supported by default.
        """
        return False



    def get_generate_stream_function(self, model, model_path: str):
        """Get the generate stream function of the model

        This method is meant to be implemented by subclasses to obtain
        the function for generating streams from the model.

        Args:
            model: The model object.
            model_path (str): The path to the model.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError



    def get_async_generate_stream_function(self, model, model_path: str):
        """Get the asynchronous generate stream function of the model

        This method is meant to be implemented by subclasses to obtain
        the function for asynchronously generating streams from the model.

        Args:
            model: The model object.
            model_path (str): The path to the model.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError



    def get_default_conv_template(
        self, model_name: str, model_path: str
    ) -> Optional[ConversationAdapter]:
        """Get the default conversation template

        This method is meant to be implemented by subclasses to obtain
        the default conversation template for a given model.

        Args:
            model_name (str): The name of the model.
            model_path (str): The path to the model.

        Returns:
            Optional[ConversationAdapter]: The conversation template, or None if not available.
        """
        raise NotImplementedError
    def get_default_message_separator(self) -> str:
        """获取默认的消息分隔符"""
        try:
            # 调用实例方法获取默认对话模板
            conv_template = self.get_default_conv_template(
                self.model_name, self.model_path
            )
            # 返回对话模板中的分隔符
            return conv_template.sep
        except Exception:
            # 如果出现异常，返回默认的换行符作为分隔符
            return "\n"

    def get_prompt_roles(self) -> List[str]:
        """获取提示信息的角色列表

        Returns:
            List[str]: 提示信息的角色列表
        """
        # 初始角色包括人类和AI
        roles = [ModelMessageRoleType.HUMAN, ModelMessageRoleType.AI]
        # 如果支持系统消息，则添加系统角色
        if self.support_system_message:
            roles.append(ModelMessageRoleType.SYSTEM)
        return roles

    def transform_model_messages(
        self, messages: List[ModelMessage], convert_to_compatible_format: bool = False
    ) -> List[Dict[str, str]]:
        """转换模型的消息格式

        默认为OpenAI格式，例如：
            .. code-block:: python

                return_messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]

        但某些模型可能需要将消息转换为其他格式（例如没有系统消息的情况），如：
            .. code-block:: python

                return_messages = [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
        
        Args:
            messages (List[ModelMessage]): 模型的消息列表
            convert_to_compatible_format (bool, optional): 是否转换为兼容格式。默认为False.

        Returns:
            List[Dict[str, str]]: 转换后的模型消息列表
        """
        # 记录日志，显示是否支持系统消息
        logger.info(f"support_system_message: {self.support_system_message}")
        # 如果不支持系统消息且需要转换为兼容格式
        if not self.support_system_message and convert_to_compatible_format:
            # 调用私有方法将消息转换为无系统消息的格式
            return self._transform_to_no_system_messages(messages)
        else:
            # 否则，调用静态方法将消息转换为通用格式
            return ModelMessage.to_common_messages(
                messages, convert_to_compatible_format=convert_to_compatible_format
            )

    def _transform_to_no_system_messages(
        self, messages: List[ModelMessage]
    ) -> List[Dict[str, str]]:
        """将消息转换为没有系统消息的格式

        Args:
            messages (List[ModelMessage]): 模型的消息列表

        Returns:
            List[Dict[str, str]]: 转换后的模型消息列表
        """
        # 实现将系统消息过滤掉的转换逻辑
        transformed_messages = [
            {"role": message.role, "content": message.content}
            for message in messages
            if message.role != ModelMessageRoleType.SYSTEM
        ]
        return transformed_messages
    ) -> List[Dict[str, str]]:
        """Transform the model messages to no system messages

        Some opensource chat model no system messages, so wo should transform the messages to no system messages.

        Merge the system messages to the last user message, example:
            .. code-block:: python

                return_messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ]
            =>
                return_messages = [
                    {"role": "user", "content": "You are a helpful assistant\nHello"},
                    {"role": "assistant", "content": "Hi"}
                ]

        Args:
            messages (List[ModelMessage]): The model messages

        Returns:
            List[Dict[str, str]]: The transformed model messages
        """
        # Transform model messages to a common format without system messages
        openai_messages = ModelMessage.to_common_messages(messages)
        # List to store system messages separately
        system_messages = []
        # List to store transformed model messages
        return_messages = []
        
        # Iterate through each message in the transformed openai_messages
        for message in openai_messages:
            # Check if the message role is "system"; if so, collect its content
            if message["role"] == "system":
                system_messages.append(message["content"])
            else:
                # If the message role is not "system", add it to return_messages
                return_messages.append(message)
        
        # If there are more than one system message, log a warning
        if len(system_messages) > 1:
            logger.warning("Your system messages have more than one message")
        
        # If there are any system messages collected
        if system_messages:
            # Get the default message separator
            sep = self.get_default_message_separator()
            # Concatenate all system messages into a single string
            str_system_messages = ",".join(system_messages)
            # Update the content of the last user message with concatenated system messages
            return_messages[-1]["content"] = (
                str_system_messages + sep + return_messages[-1]["content"]
            )
        
        # Return the transformed model messages
        return return_messages

    def get_str_prompt(
        self,
        params: Dict,
        messages: List[ModelMessage],
        tokenizer: Any,
        prompt_template: str = None,
        convert_to_compatible_format: bool = False,
    ) -> Optional[str]:
        """Get the string prompt from the given parameters and messages

        If the value of return is not None, we will skip :meth:`~LLMModelAdapter.get_prompt_with_template` and use the value of return.

        Args:
            params (Dict): The parameters
            messages (List[ModelMessage]): The model messages
            tokenizer (Any): The tokenizer of model, in huggingface chat model, we can create the prompt by tokenizer
            prompt_template (str, optional): The prompt template. Defaults to None.
            convert_to_compatible_format (bool, optional): Whether to convert to compatible format. Defaults to False.

        Returns:
            Optional[str]: The string prompt
        """
        # This method currently always returns None
        return None
    def get_prompt_with_template(
        self,
        params: Dict,
        messages: List[ModelMessage],
        model_name: str,
        model_path: str,
        model_context: Dict,
        prompt_template: str = None,
    ):
        # 获取参数中的 `convert_to_compatible_format` 字段
        convert_to_compatible_format = params.get("convert_to_compatible_format")
        # 调用 `get_default_conv_template` 方法获取 ConversationAdapter 对象
        conv: ConversationAdapter = self.get_default_conv_template(
            model_name, model_path
        )

        if prompt_template:
            # 如果存在 prompt_template 参数，则使用配置中的模板
            logger.info(f"Use prompt template {prompt_template} from config")
            conv = get_conv_template(prompt_template)
        if not conv or not messages:
            # 如果 conv 不存在或者 messages 列表为空，则记录日志并返回空值
            logger.info(
                f"No conv from model_path {model_path} or no messages in params, {self}"
            )
            return None, None, None

        conv = conv.copy()
        if convert_to_compatible_format:
            # 如果需要转换为兼容格式，则调用 `_set_conv_converted_messages` 方法
            conv = self._set_conv_converted_messages(conv, messages)
        else:
            # 否则直接使用 messages
            conv = self._set_conv_messages(conv, messages)

        # 为助手添加一个空消息
        conv.append_message(conv.roles[1], None)
        # 获取更新后的 prompt
        new_prompt = conv.get_prompt()
        return new_prompt, conv.stop_str, conv.stop_token_ids

    def _set_conv_messages(
        self, conv: ConversationAdapter, messages: List[ModelMessage]
    ) -> ConversationAdapter:
        """Set the messages to the conversation template

        Args:
            conv (ConversationAdapter): The conversation template
            messages (List[ModelMessage]): The model messages

        Returns:
            ConversationAdapter: The conversation template with messages
        """
        system_messages = []
        for message in messages:
            if isinstance(message, ModelMessage):
                role = message.role
                content = message.content
            elif isinstance(message, dict):
                role = message["role"]
                content = message["content"]
            else:
                # 抛出异常，如果消息类型既不是 ModelMessage 也不是 dict
                raise ValueError(f"Invalid message type: {message}")

            if role == ModelMessageRoleType.SYSTEM:
                system_messages.append(content)
            elif role == ModelMessageRoleType.HUMAN:
                # 将人类角色的消息添加到对话模板中的第一个角色
                conv.append_message(conv.roles[0], content)
            elif role == ModelMessageRoleType.AI:
                # 将 AI 角色的消息添加到对话模板中的第二个角色
                conv.append_message(conv.roles[1], content)
            else:
                # 抛出异常，如果角色类型未知
                raise ValueError(f"Unknown role: {role}")
        if len(system_messages) > 1:
            # 如果系统消息超过一个，则抛出异常
            raise ValueError(
                f"Your system messages have more than one message: {system_messages}"
            )
        if system_messages:
            # 设置系统消息
            conv.set_system_message(system_messages[0])
        return conv

    def _set_conv_converted_messages(
        self, conv: ConversationAdapter, messages: List[ModelMessage]
    ) -> ConversationAdapter:
        """Set the converted messages to the conversation template

        Args:
            conv (ConversationAdapter): The conversation template
            messages (List[ModelMessage]): The model messages

        Returns:
            ConversationAdapter: The conversation template with converted messages
        """
        system_messages = []
        for message in messages:
            if isinstance(message, ModelMessage):
                role = message.role
                content = message.content
            elif isinstance(message, dict):
                role = message["role"]
                content = message["content"]
            else:
                # 抛出异常，如果消息类型既不是 ModelMessage 也不是 dict
                raise ValueError(f"Invalid message type: {message}")

            if role == ModelMessageRoleType.SYSTEM:
                system_messages.append(content)
            elif role == ModelMessageRoleType.HUMAN:
                # 将人类角色的消息转换并添加到对话模板中的第一个角色
                conv.append_message(conv.roles[0], self._convert_message(content))
            elif role == ModelMessageRoleType.AI:
                # 将 AI 角色的消息转换并添加到对话模板中的第二个角色
                conv.append_message(conv.roles[1], self._convert_message(content))
            else:
                # 抛出异常，如果角色类型未知
                raise ValueError(f"Unknown role: {role}")
        if len(system_messages) > 1:
            # 如果系统消息超过一个，则抛出异常
            raise ValueError(
                f"Your system messages have more than one message: {system_messages}"
            )
        if system_messages:
            # 设置系统消息
            conv.set_system_message(system_messages[0])
        return conv

    def _convert_message(self, content: str) -> str:
        """Convert message content to compatible format if needed

        Args:
            content (str): The message content

        Returns:
            str: Converted message content
        """
        # 在需要时，对消息内容进行格式转换的具体实现
        # 这里假设实现了一个将消息内容转换为兼容格式的函数或方法
        return content
    ) -> ConversationAdapter:
        """Set the messages to the conversation template

        In the old version, we will convert the messages to compatible format.
        This method will be deprecated in the future.

        Args:
            conv (ConversationAdapter): The conversation template
            messages (List[ModelMessage]): The model messages

        Returns:
            ConversationAdapter: The conversation template with messages
        """
        # Initialize empty lists to categorize messages
        system_messages = []  # List to store system messages
        user_messages = []    # List to store user messages
        ai_messages = []      # List to store AI messages

        # Iterate through each message in the input messages list
        for message in messages:
            if isinstance(message, ModelMessage):
                role = message.role         # Extract role from ModelMessage object
                content = message.content   # Extract content from ModelMessage object
            elif isinstance(message, dict):
                role = message["role"]      # Extract role from dictionary
                content = message["content"]    # Extract content from dictionary
            else:
                raise ValueError(f"Invalid message type: {message}")  # Raise error for unsupported message type

            # Categorize messages based on role type
            if role == ModelMessageRoleType.SYSTEM:
                system_messages.append(content)    # Add system message to system_messages list
            elif role == ModelMessageRoleType.HUMAN:
                user_messages.append(content)      # Add user message to user_messages list
            elif role == ModelMessageRoleType.AI:
                ai_messages.append(content)        # Add AI message to ai_messages list
            else:
                raise ValueError(f"Unknown role: {role}")   # Raise error for unknown role type

        # Initialize a list to store system messages that can be used
        can_use_systems: [] = []

        # Process system messages
        if system_messages:
            if len(system_messages) > 1:
                # Handle multiple system messages scenario
                user_messages[-1] = system_messages[-1]  # Replace the last user message with the last system message
                can_use_systems = system_messages[:-1]   # Store all but the last system message
            else:
                can_use_systems = system_messages   # Store the single system message

        # Add messages to the conversation template
        for i in range(len(user_messages)):
            conv.append_message(conv.roles[0], user_messages[i])   # Append user messages to the conversation template
            if i < len(ai_messages):
                conv.append_message(conv.roles[1], ai_messages[i])    # Append AI messages to the conversation template

        # Set system messages in the conversation template
        conv.set_system_message("".join(can_use_systems))    # Join system messages and set in the conversation template

        return conv    # Return the updated conversation template

    def apply_conv_template(self) -> bool:
        return self.model_type() != ModelType.PROXY   # Check if model type is not ModelType.PROXY

    def model_adaptation(
        self,
        params: Dict,
        model_name: str,
        model_path: str,
        tokenizer: Any,
        prompt_template: str = None,
    ) -> Tuple[Dict, Dict]:
        """Params adaptation"""
        # 从参数中获取消息列表
        messages = params.get("messages")
        # 从参数中获取是否需要转换为兼容格式的标志
        convert_to_compatible_format = params.get("convert_to_compatible_format")
        # 从参数中获取消息版本，若未指定则默认为 "v2"，并转换为小写
        message_version = params.get("version", "v2").lower()
        # 记录日志，显示消息版本信息
        logger.info(f"Message version is {message_version}")
        # 若未指定是否转换兼容格式，则根据消息版本决定是否转换为 v1 格式
        if convert_to_compatible_format is None:
            # 当消息版本为 "v1" 时，支持将消息转换为兼容格式
            convert_to_compatible_format = message_version == "v1"
        # 将确定的转换兼容格式标志保存到参数中
        params["convert_to_compatible_format"] = convert_to_compatible_format

        # 初始化模型上下文信息
        model_context = {
            "prompt_echo_len_char": -1,
            "has_format_prompt": False,
            "echo": params.get("echo", True),
        }
        # 如果存在消息列表，则将每个消息转换为 ModelMessage 类型
        if messages:
            messages = [
                m if isinstance(m, ModelMessage) else ModelMessage(**m)
                for m in messages
            ]
            # 将转换后的消息列表保存到参数中
            params["messages"] = messages
        # 将消息列表转换为字符串格式，并保存到参数中的 "string_prompt" 键下
        params["string_prompt"] = ModelMessage.messages_to_string(messages)

        # 如果不需要应用会话模板，则直接返回当前的参数和模型上下文
        if not self.apply_conv_template():
            # 不需要应用会话模板，此时适用于代理 LLM
            return params, model_context

        # 根据参数和消息等信息获取新的提示信息
        new_prompt = self.get_str_prompt(
            params, messages, tokenizer, prompt_template, convert_to_compatible_format
        )
        conv_stop_str, conv_stop_token_ids = None, None
        # 如果未获取到新的提示信息，则尝试使用模板生成新的提示信息
        if not new_prompt:
            (
                new_prompt,
                conv_stop_str,
                conv_stop_token_ids,
            ) = self.get_prompt_with_template(
                params, messages, model_name, model_path, model_context, prompt_template
            )
            # 如果仍未获取到新的提示信息，则直接返回当前的参数和模型上下文
            if not new_prompt:
                return params, model_context

        # 计算新提示信息中去除特定标记后的字符长度，并保存到模型上下文中
        prompt_echo_len_char = len(new_prompt.replace("</s>", "").replace("<s>", ""))
        model_context["prompt_echo_len_char"] = prompt_echo_len_char
        # 标记模型上下文中的提示格式已设置
        model_context["has_format_prompt"] = True
        # 将新的提示信息保存到参数中的 "prompt" 键下
        params["prompt"] = new_prompt

        # 获取自定义停止标记及其对应的停止标记 ID
        custom_stop = params.get("stop")
        custom_stop_token_ids = params.get("stop_token_ids")

        # 使用输入参数中传递的值优先设置停止标记及其对应的停止标记 ID
        params["stop"] = custom_stop or conv_stop_str
        params["stop_token_ids"] = custom_stop_token_ids or conv_stop_token_ids

        # 返回更新后的参数和模型上下文
        return params, model_context
class AdapterEntry:
    """The entry of model adapter"""

    def __init__(
        self,
        model_adapter: LLMModelAdapter,
        match_funcs: List[Callable[[str, str, str], bool]] = None,
    ):
        # 初始化函数，接受一个模型适配器和匹配函数列表作为参数
        self.model_adapter = model_adapter
        self.match_funcs = match_funcs or []


model_adapters: List[AdapterEntry] = []


def register_model_adapter(
    model_adapter_cls: Type[LLMModelAdapter],
    match_funcs: List[Callable[[str, str, str], bool]] = None,
) -> None:
    """Register a model adapter.

    Args:
        model_adapter_cls (Type[LLMModelAdapter]): The model adapter class.
        match_funcs (List[Callable[[str, str, str], bool]], optional): The match functions. Defaults to None.
    """
    # 注册一个模型适配器
    model_adapters.append(AdapterEntry(model_adapter_cls(), match_funcs))


def get_model_adapter(
    model_type: str,
    model_name: str,
    model_path: str,
    conv_factory: Optional[ConversationAdapterFactory] = None,
) -> Optional[LLMModelAdapter]:
    """Get a model adapter.

    Args:
        model_type (str): The type of the model.
        model_name (str): The name of the model.
        model_path (str): The path of the model.
        conv_factory (Optional[ConversationAdapterFactory], optional): The conversation factory. Defaults to None.
    Returns:
        Optional[LLMModelAdapter]: The model adapter.
    """
    adapter = None
    # 首先通过模型名称查找适配器
    for adapter_entry in model_adapters[::-1]:
        if adapter_entry.model_adapter.match(model_type, model_name, None):
            adapter = adapter_entry.model_adapter
            break
    for adapter_entry in model_adapters[::-1]:
        if adapter_entry.model_adapter.match(model_type, None, model_path):
            adapter = adapter_entry.model_adapter
            break
    if adapter:
        new_adapter = adapter.new_adapter()
        new_adapter.model_name = model_name
        new_adapter.model_path = model_path
        if conv_factory:
            new_adapter.conv_factory = conv_factory
        return new_adapter
    return None
```