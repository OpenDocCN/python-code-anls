# `.\AutoGPT\autogpts\autogpt\autogpt\models\base_open_ai_plugin.py`

```py
"""Handles loading of plugins."""
# 导入必要的模块和类型定义
from typing import Any, Dict, List, Optional, Tuple, TypedDict, TypeVar

# 导入 AutoGPTPluginTemplate 类
from auto_gpt_plugin_template import AutoGPTPluginTemplate

# 定义 TypeVar 类型变量 PromptGenerator
PromptGenerator = TypeVar("PromptGenerator")

# 定义 TypedDict 类型 Message
class Message(TypedDict):
    role: str
    content: str

# 定义 BaseOpenAIPlugin 类，继承自 AutoGPTPluginTemplate
class BaseOpenAIPlugin(AutoGPTPluginTemplate):
    """
    This is a BaseOpenAIPlugin class for generating AutoGPT plugins.
    """

    # 初始化方法，接受 manifests_specs_clients 参数
    def __init__(self, manifests_specs_clients: dict):
        # 初始化插件的名称、版本、描述、客户端、清单和 OpenAPI 规范
        self._name = manifests_specs_clients["manifest"]["name_for_model"]
        self._version = manifests_specs_clients["manifest"]["schema_version"]
        self._description = manifests_specs_clients["manifest"]["description_for_model"]
        self._client = manifests_specs_clients["client"]
        self._manifest = manifests_specs_clients["manifest"]
        self._openapi_spec = manifests_specs_clients["openapi_spec"]

    # 检查插件是否能处理 on_response 方法
    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.
        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    # 处理从模型接收到的响应
    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        return response

    # 检查插件是否能处理 post_prompt 方法
    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.
        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return False

    # 在 generate_prompt 调用后、生成提示之前调用的方法
    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.
        Args:
            prompt (PromptGenerator): The prompt generator.
        Returns:
            PromptGenerator: The prompt generator.
        """
        return prompt
    def can_handle_on_planning(self) -> bool:
        """检查插件是否能处理 on_planning 方法。
        返回:
            bool: 如果插件能处理 on_planning 方法则返回 True。"""
        return False

    def on_planning(
        self, prompt: PromptGenerator, messages: List[Message]
    ) -> Optional[str]:
        """在规划聊天完成之前调用此方法。
        参数:
            prompt (PromptGenerator): 提示生成器。
            messages (List[str]): 消息列表。"""

    def can_handle_post_planning(self) -> bool:
        """检查插件是否能处理 post_planning 方法。
        返回:
            bool: 如果插件能处理 post_planning 方法则返回 True。"""
        return False

    def post_planning(self, response: str) -> str:
        """在规划聊天完成之后调用此方法。
        参数:
            response (str): 响应。
        返回:
            str: 处理后的响应。"""
        return response

    def can_handle_pre_instruction(self) -> bool:
        """检查插件是否能处理 pre_instruction 方法。
        返回:
            bool: 如果插件能处理 pre_instruction 方法则返回 True。"""
        return False

    def pre_instruction(self, messages: List[Message]) -> List[Message]:
        """在指令聊天完成之前调用此方法。
        参数:
            messages (List[Message]): 上下文消息列表。
        返回:
            List[Message]: 处理后的消息列表。"""
        return messages
    # 检查插件是否能处理 on_instruction 方法
    def can_handle_on_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_instruction method.
        Returns:
            bool: True if the plugin can handle the on_instruction method."""
        return False

    # 当指令聊天结束时调用此方法
    def on_instruction(self, messages: List[Message]) -> Optional[str]:
        """This method is called when the instruction chat is done.
        Args:
            messages (List[Message]): The list of context messages.
        Returns:
            Optional[str]: The resulting message.
        """

    # 检查插件是否能处理 post_instruction 方法
    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.
        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    # 指令聊天结束后调用此方法
    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.
        Args:
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        return response

    # 检查插件是否能处理 pre_command 方法
    def can_handle_pre_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_command method.
        Returns:
            bool: True if the plugin can handle the pre_command method."""
        return False

    # 在执行命令之前调用此方法
    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """This method is called before the command is executed.
        Args:
            command_name (str): The command name.
            arguments (Dict[str, Any]): The arguments.
        Returns:
            Tuple[str, Dict[str, Any]]: The command name and the arguments.
        """
        return command_name, arguments
    # 检查插件是否能处理 post_command 方法
    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.
        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    # 在命令执行后调用此方法
    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.
        Args:
            command_name (str): The command name.
            response (str): The response.
        Returns:
            str: The resulting response.
        """
        return response

    # 检查插件是否能处理 chat_completion 方法
    def can_handle_chat_completion(
        self, messages: Dict[Any, Any], model: str, temperature: float, max_tokens: int
    ) -> bool:
        """This method is called to check that the plugin can
          handle the chat_completion method.
        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
          Returns:
              bool: True if the plugin can handle the chat_completion method."""
        return False

    # 在聊天完成后调用此方法
    def handle_chat_completion(
        self, messages: List[Message], model: str, temperature: float, max_tokens: int
    ) -> str:
        """This method is called when the chat completion is done.
        Args:
            messages (List[Message]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.
        Returns:
            str: The resulting response.
        """
    # 检查插件是否能处理文本嵌入的方法
    def can_handle_text_embedding(self, text: str) -> bool:
        """This method is called to check that the plugin can
          handle the text_embedding method.

        Args:
            text (str): The text to be convert to embedding.
        Returns:
            bool: True if the plugin can handle the text_embedding method."""
        return False

    # 创建文本嵌入的方法
    def handle_text_embedding(self, text: str) -> list[float]:
        """This method is called to create a text embedding.

        Args:
            text (str): The text to be convert to embedding.
        Returns:
            list[float]: The created embedding vector.
        """

    # 检查插件是否能处理用户输入的方法
    def can_handle_user_input(self, user_input: str) -> bool:
        """This method is called to check that the plugin can
        handle the user_input method.

        Args:
            user_input (str): The user input.

        Returns:
            bool: True if the plugin can handle the user_input method."""
        return False

    # 请求用户输入的方法
    def user_input(self, user_input: str) -> str:
        """This method is called to request user input to the user.

        Args:
            user_input (str): The question or prompt to ask the user.

        Returns:
            str: The user input.
        """

    # 检查插件是否能处理报告的方法
    def can_handle_report(self) -> bool:
        """This method is called to check that the plugin can
        handle the report method.

        Returns:
            bool: True if the plugin can handle the report method."""
        return False

    # 向用户报告消息的方法
    def report(self, message: str) -> None:
        """This method is called to report a message to the user.

        Args:
            message (str): The message to report.
        """
```