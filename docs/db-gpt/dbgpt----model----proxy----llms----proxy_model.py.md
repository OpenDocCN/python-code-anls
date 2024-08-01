# `.\DB-GPT-src\dbgpt\model\proxy\llms\proxy_model.py`

```py
from __future__ import annotations
# 导入必要的模块和类
import logging
from typing import TYPE_CHECKING, List, Optional, Union

from dbgpt.model.parameter import ProxyModelParameters  # 导入ProxyModelParameters类
from dbgpt.model.proxy.base import ProxyLLMClient  # 导入ProxyLLMClient类
from dbgpt.model.utils.token_utils import ProxyTokenizerWrapper  # 导入ProxyTokenizerWrapper类

if TYPE_CHECKING:
    from dbgpt.core.interface.message import BaseMessage, ModelMessage  # 导入BaseMessage和ModelMessage类

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ProxyModel:
    def __init__(
        self,
        model_params: ProxyModelParameters,  # 初始化ProxyModel对象的模型参数
        proxy_llm_client: Optional[ProxyLLMClient] = None,
    ) -> None:
        self._model_params = model_params  # 设置ProxyModel对象的模型参数
        self._tokenizer = ProxyTokenizerWrapper()  # 使用ProxyTokenizerWrapper创建代理模型的标记器
        self.proxy_llm_client = proxy_llm_client  # 设置代理LLM客户端对象

    def get_params(self) -> ProxyModelParameters:
        return self._model_params  # 返回ProxyModel对象的模型参数

    def count_token(
        self,
        messages: Union[str, BaseMessage, ModelMessage, List[ModelMessage]],  # 接受不同类型消息的列表或单个消息来计算令牌数
        model_name: Optional[int] = None,  # 可选的模型名称，默认为None
    ) -> int:
        """Count token of given messages

        Args:
            messages (Union[str, BaseMessage, ModelMessage, List[ModelMessage]]): messages to count token
            model_name (Optional[int], optional): model name. Defaults to None.

        Returns:
            int: token count, -1 if failed
        """
        return self._tokenizer.count_token(messages, model_name)  # 使用标记器对象计算消息中的令牌数
```