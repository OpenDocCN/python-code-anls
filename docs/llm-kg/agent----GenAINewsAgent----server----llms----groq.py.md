# `.\agent\GenAINewsAgent\server\llms\groq.py`

```
# 从 groq 模块导入 Groq 和 AsyncGroq 类
from groq import Groq, AsyncGroq
# 导入用于异常处理的 traceback 模块
import traceback
# 导入类型提示模块 List、Dict 和 Union
from typing import List, Dict, Union
# 导入基础 LLM 功能的 BaseLLM 类
from llms.base import BaseLLM
# 导入上下文管理模块 ContextManagement
from llms.ctx import ContextManagement
# 从 groq 模块导入 RateLimitError 异常类
from groq import RateLimitError
# 导入 backoff 模块，用于实现指数退避策略
import backoff

# 使用 ContextManagement 类创建上下文管理对象
manageContext = ContextManagement()

# 继承自 BaseLLM 类的 GroqLLM 类
class GroqLLM(BaseLLM):

    def __init__(self, api_key: Union[str, None] = None):
        # 调用父类 BaseLLM 的构造函数
        super().__init__(api_key)
        # 初始化 AsyncGroq 客户端对象
        self.client = AsyncGroq(api_key=api_key)

    # 使用 backoff 库的指数退避策略处理 RateLimitError 异常，最多重试 3 次
    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=3)
    async def __call__(self, model: str, messages: List[Dict], **kwargs):
        try:
            # 如果 kwargs 中包含 "system" 键
            if "system" in kwargs:
                # 将系统消息加入到 messages 列表中
                messages = [{
                    "role": "system",
                    "content": kwargs.get("system")
                }] + messages
                # 删除 kwargs 中的 "system" 键
                del kwargs["system"]
            # 如果 kwargs 中包含 "ctx_length" 键
            if "ctx_length" in kwargs:
                # 删除 kwargs 中的 "ctx_length" 键
                del kwargs["ctx_length"]
            # 调用 manageContext 函数处理消息列表
            messages = manageContext(messages, kwargs.get("ctx_length", 7_000))
            # 调用 AsyncGroq 客户端的创建聊天完成任务方法，获取输出结果
            output = await self.client.chat.completions.create(
                messages=messages, model=model, **kwargs)
            # 返回输出结果中的第一个选择消息内容
            return output.choices[0].message.content
        # 捕获 RateLimitError 异常
        except RateLimitError:
            raise RateLimitError
        # 捕获所有其他异常并打印错误信息
        except Exception as err:
            print(f"ERROR: {str(err)}")
            print(f"{traceback.format_exc()}")
            # 返回空字符串
            return ""


# 继承自 BaseLLM 类的 GroqLLMStream 类
class GroqLLMStream(BaseLLM):

    def __init__(self, api_key: Union[str, None] = None):
        # 调用父类 BaseLLM 的构造函数
        super().__init__(api_key)
        # 初始化 AsyncGroq 客户端对象
        self.client = AsyncGroq(api_key=api_key)

    async def __call__(self, model: str, messages: List[Dict], **kwargs):
        # 如果 kwargs 中包含 "system" 键
        if "system" in kwargs:
            # 将系统消息加入到 messages 列表中
            messages = [{
                "role": "system",
                "content": kwargs.get("system")
            }] + messages
            # 删除 kwargs 中的 "system" 键
            del kwargs["system"]
        # 如果 kwargs 中包含 "ctx_length" 键
        if "ctx_length" in kwargs:
            # 删除 kwargs 中的 "ctx_length" 键
            del kwargs["ctx_length"]
        # 调用 manageContext 函数处理消息列表
        messages = manageContext(messages, kwargs.get("ctx_length", 7_000))
        # 调用 AsyncGroq 客户端的创建聊天完成任务方法，获取输出结果（流式输出）
        output = await self.client.chat.completions.create(
            messages=messages,
            model=model,
            stream=True,  # 使用流式输出
            **kwargs
        )
        # 异步迭代处理输出的每个 chunk
        async for chunk in output:
            # 返回 chunk 中的第一个选择消息内容的 delta 字段，如果没有则返回空字符串
            yield chunk.choices[0].delta.content or ""
```