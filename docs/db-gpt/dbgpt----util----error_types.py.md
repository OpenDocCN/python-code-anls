# `.\DB-GPT-src\dbgpt\util\error_types.py`

```py
class LLMChatError(Exception):
    """
    定义了一个名为 LLMChatError 的异常类，继承自内置的 Exception 类。
    用于表示在 LLM 聊天生成过程中可能发生的异常情况。
    """

    def __init__(self, message="LLM Chat Generrate Error!", original_exception=None):
        # 调用父类 Exception 的初始化方法，传入异常消息 message
        super().__init__(message)
        # 设置异常对象的 message 属性为传入的异常消息
        self.message = message
        # 设置异常对象的 original_exception 属性为传入的原始异常对象
        self.original_exception = original_exception

    def __str__(self):
        if self.original_exception:
            # 如果存在原始异常对象，则返回带有自定义异常消息和原始异常消息的字符串表示
            return f"{self.message}({self.original_exception})"
        else:
            # 如果不存在原始异常对象，则返回自定义异常消息的字符串表示
            return self.message
```