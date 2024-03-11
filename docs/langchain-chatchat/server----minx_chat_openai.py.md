# `.\Langchain-Chatchat\server\minx_chat_openai.py`

```py
# 导入必要的模块和类型
from typing import (
    TYPE_CHECKING,
    Any,
    Tuple
)
import sys
import logging

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 如果是类型检查，则导入tiktoken模块
if TYPE_CHECKING:
    import tiktoken

# 定义MinxChatOpenAI类
class MinxChatOpenAI:

    # 静态方法：导入tiktoken模块
    @staticmethod
    def import_tiktoken() -> Any:
        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to calculate get_token_ids. "
                "Please install it with `pip install tiktoken`."
            )
        return tiktoken

    # 静态方法：获取编码模型
    @staticmethod
    def get_encoding_model(self) -> Tuple[str, "tiktoken.Encoding"]:
        # 导入tiktoken模块
        tiktoken_ = MinxChatOpenAI.import_tiktoken()
        # 根据条件选择模型名称
        if self.tiktoken_model_name is not None:
            model = self.tiktoken_model_name
        else:
            model = self.model_name
            if model == "gpt-3.5-turbo":
                # gpt-3.5-turbo可能会随时间变化而改变
                # 假设使用gpt-3.5-turbo-0301返回令牌数量
                model = "gpt-3.5-turbo-0301"
            elif model == "gpt-4":
                # gpt-4可能会随时间变化而改变
                # 假设使用gpt-4-0314返回令牌数量
                model = "gpt-4-0314"
        # 尝试获取模型的编码
        try:
            encoding = tiktoken_.encoding_for_model(model)
        except Exception as e:
            logger.warning("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken_.get_encoding(model)
        return model, encoding
```