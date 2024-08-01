# `.\DB-GPT-src\dbgpt\model\utils\token_utils.py`

```py
# 从未来模块中导入注释，这使得该文件在旧版Python中也可以运行
from __future__ import annotations

# 导入日志模块
import logging
# 导入类型检查相关的模块
from typing import TYPE_CHECKING, List, Optional, Union

# 如果处于类型检查状态，导入特定模块
if TYPE_CHECKING:
    from dbgpt.core.interface.message import BaseMessage, ModelMessage

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


# 定义代理标记器包装类
class ProxyTokenizerWrapper:
    def __init__(self) -> None:
        # 初始化支持编码的状态为True
        self._support_encoding = True
        # 初始化编码模型为None
        self._encoding_model = None

    def count_token(
        self,
        messages: Union[str, BaseMessage, ModelMessage, List[ModelMessage]],
        model_name: Optional[str] = None,
    ) -> int:
        """Count token of given messages

        Args:
            messages (Union[str, BaseMessage, ModelMessage, List[ModelMessage]]): messages to count token
            model_name (Optional[str], optional): model name. Defaults to None.

        Returns:
            int: token count, -1 if failed
        """
        # 如果不支持编码，记录警告信息并返回-1
        if not self._support_encoding:
            logger.warning(
                "model does not support encoding model, can't count token, returning -1"
            )
            return -1
        
        # 获取或创建指定的编码模型
        encoding = self._get_or_create_encoding_model(model_name)
        cnt = 0
        
        # 根据不同类型的消息计算token数
        if isinstance(messages, str):
            cnt = len(encoding.encode(messages, disallowed_special=()))
        elif isinstance(messages, BaseMessage):
            cnt = len(encoding.encode(messages.content, disallowed_special=()))
        elif isinstance(messages, ModelMessage):
            cnt = len(encoding.encode(messages.content, disallowed_special=()))
        elif isinstance(messages, list):
            for message in messages:
                cnt += len(encoding.encode(message.content, disallowed_special=()))
        else:
            # 如果消息类型不支持，记录警告信息并返回-1
            logger.warning(
                "unsupported type of messages, can't count token, returning -1"
            )
            return -1
        
        return cnt
    def _get_or_create_encoding_model(self, model_name: Optional[str] = None):
        """获取或创建给定模型名称的编码模型

        如果已经存在编码模型，则直接返回它；否则尝试根据给定的模型名称创建一个新的编码模型。

        更多细节参见：https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        # 如果已经存在编码模型，直接返回
        if self._encoding_model:
            return self._encoding_model

        try:
            import tiktoken

            # 如果导入了tiktoken，说明可以使用它来计算token数量
            logger.info(
                "tiktoken 已安装，将用它来计算token数量。tiktoken 将会从网络下载tokenizer，"
                "也可以手动下载并放置在环境变量 TIKTOKEN_CACHE_DIR 指定的目录中。"
            )
        except ImportError:
            # 如果导入失败，标记不支持编码，并记录警告信息
            self._support_encoding = False
            logger.warn("未安装 tiktoken，无法计算token数量，返回 -1")
            return -1

        try:
            # 如果未指定模型名称，则使用默认名称 "gpt-3.5-turbo"
            if not model_name:
                model_name = "gpt-3.5-turbo"

            # 根据模型名称获取或创建编码模型
            self._encoding_model = tiktoken.model.encoding_for_model(model_name)
        except KeyError:
            # 如果指定模型名称的tokenizer未找到，则使用默认的 "cl100k_base" 编码
            logger.warning(
                f"{model_name} 的tokenizer未找到，将使用 cl100k_base 编码。"
            )
            self._encoding_model = tiktoken.get_encoding("cl100k_base")

        # 返回获取或创建的编码模型
        return self._encoding_model
```