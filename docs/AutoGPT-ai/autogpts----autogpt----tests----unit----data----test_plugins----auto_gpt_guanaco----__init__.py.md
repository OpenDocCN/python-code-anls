# `.\AutoGPT\autogpts\autogpt\tests\unit\data\test_plugins\auto_gpt_guanaco\__init__.py`

```py
"""This is the Test plugin for AutoGPT."""
# 导入必要的模块和类型
from typing import Any, Dict, List, Optional, Tuple, TypeVar

# 导入 AutoGPTPluginTemplate 类
from auto_gpt_plugin_template import AutoGPTPluginTemplate

# 定义一个类型变量 PromptGenerator
PromptGenerator = TypeVar("PromptGenerator")


# 定义 AutoGPTGuanaco 类，继承自 AutoGPTPluginTemplate
class AutoGPTGuanaco(AutoGPTPluginTemplate):
    """
    This is plugin for AutoGPT.
    """

    # 初始化方法
    def __init__(self):
        super().__init__()
        self._name = "AutoGPT-Guanaco"
        self._version = "0.1.0"
        self._description = "This is a Guanaco local model plugin."

    # 检查是否能处理 on_response 方法
    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.

        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    # 处理响应的方法
    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        if len(response):
            print("OMG OMG It's Alive!")
        else:
            print("Is it alive?")

    # 检查是否能处理 post_prompt 方法
    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.

        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return False

    # 处理生成提示后的方法
    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.

        Args:
            prompt (PromptGenerator): The prompt generator.

        Returns:
            PromptGenerator: The prompt generator.
        """

    # 检查是否能处理 on_planning 方法
    def can_handle_on_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_planning method.

        Returns:
            bool: True if the plugin can handle the on_planning method."""
        return False

    # 处理规划的方法
    def on_planning(
        self, prompt: PromptGenerator, messages: List[str]
    def can_handle_post_planning(self) -> bool:
        """检查插件是否能处理 post_planning 方法。

        Returns:
            bool: 如果插件能处理 post_planning 方法，则返回 True。
        """
        return False

    def post_planning(self, response: str) -> str:
        """在规划聊天完成后调用此方法。

        Args:
            response (str): 响应。

        Returns:
            str: 结果响应。
        """

    def can_handle_pre_instruction(self) -> bool:
        """检查插件是否能处理 pre_instruction 方法。

        Returns:
            bool: 如果插件能处理 pre_instruction 方法，则返回 True。
        """
        return False

    def pre_instruction(self, messages: List[str]) -> List[str]:
        """在指令聊天完成前调用此方法。

        Args:
            messages (List[str]): 上下文消息列表。

        Returns:
            List[str]: 结果消息列表。
        """

    def can_handle_on_instruction(self) -> bool:
        """检查插件是否能处理 on_instruction 方法。

        Returns:
            bool: 如果插件能处理 on_instruction 方法，则返回 True。
        """
        return False

    def on_instruction(self, messages: List[str]) -> Optional[str]:
        """在指令聊天完成时调用此方法。

        Args:
            messages (List[str]): 上下文消息列表。

        Returns:
            Optional[str]: 结果消息。
        """
    # 检查插件是否能处理 post_instruction 方法
    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.

        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    # 在指令聊天结束后调用此方法
    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.

        Args:
            response (str): The response.

        Returns:
            str: The resulting response.
        """

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

    # 检查插件是否能处理 post_command 方法
    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.

        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    # 在执行命令之后调用此方法
    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.

        Args:
            command_name (str): The command name.
            response (str): The response.

        Returns:
            str: The resulting response.
        """

    # 检查插件是否能处理 chat_completion 方法
    def can_handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    def can_handle_chat_completion(
        self,
        messages: Dict[Any, Any],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        """This method is called to check that the plugin can
          handle the chat_completion method.

        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

        Returns:
            bool: True if the plugin can handle the chat_completion method."""
        return False

    def handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """This method is called when the chat completion is done.

        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

        Returns:
            str: The resulting response.
        """

    def can_handle_text_embedding(self, text: str) -> bool:
        """This method is called to check that the plugin can
          handle the text_embedding method.
        Args:
            text (str): The text to be convert to embedding.
        Returns:
            bool: True if the plugin can handle the text_embedding method."""
        return False

    def handle_text_embedding(self, text: str) -> list:
        """This method is called when the chat completion is done.
        Args:
            text (str): The text to be convert to embedding.
        Returns:
            list: The text embedding.
        """

    def can_handle_user_input(self, user_input: str) -> bool:
        """This method is called to check that the plugin can
        handle the user_input method.

        Args:
            user_input (str): The user input.

        Returns:
            bool: True if the plugin can handle the user_input method."""
        return False
    # 用户输入方法，用于请求用户输入
    def user_input(self, user_input: str) -> str:
        """This method is called to request user input to the user.

        Args:
            user_input (str): The question or prompt to ask the user.

        Returns:
            str: The user input.
        """

    # 检查插件是否能处理报告方法
    def can_handle_report(self) -> bool:
        """This method is called to check that the plugin can
        handle the report method.

        Returns:
            bool: True if the plugin can handle the report method."""
        return False

    # 报告方法，用于向用户报告消息
    def report(self, message: str) -> None:
        """This method is called to report a message to the user.

        Args:
            message (str): The message to report.
        """
```