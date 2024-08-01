# `.\DB-GPT-src\dbgpt\model\llm\llm_utils.py`

```py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入标准库 abc，用于定义抽象基类
import abc
# 导入 functools 库，用于创建装饰器函数
import functools
# 导入 time 库，用于处理时间相关操作

# TODO Rewrite this
# 定义 retry_stream_api 函数，用于重试 Vicuna 服务器调用
def retry_stream_api(
    num_retries: int = 10, backoff_base: float = 2.0, warn_user: bool = True
):
    """Retry an Vicuna Server call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """
    # 定义达到重试上限的消息
    retry_limit_msg = f"Error: Reached rate limit, passing..."
    # 定义指数退避时的消息模板
    backoff_msg = f"Error: API Bad gateway. Waiting {{backoff}} seconds..."

    # 定义装饰器函数 _wrapper
    def _wrapper(func):
        # 装饰器函数内部定义，保留被装饰函数的元信息
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            # 是否已经警告用户，默认为未警告
            user_warned = not warn_user
            # 计算重试次数，包括第一次尝试
            num_attempts = num_retries + 1
            # 循环执行重试
            for attempt in range(1, num_attempts + 1):
                try:
                    # 调用被装饰的函数
                    return func(*args, **kwargs)
                except Exception as e:
                    # 如果出现 HTTP 状态码不是 502 或达到最大重试次数，则抛出异常
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise

                # 计算指数退避时间
                backoff = backoff_base ** (attempt + 2)
                # 等待指数退避时间
                time.sleep(backoff)

        return _wrapped

    return _wrapper


# 定义抽象基类 ChatIO
class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str) -> str:
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream, skip_echo_len: int):
        """Stream output."""


# 定义 SimpleChatIO 类，继承自 ChatIO
class SimpleChatIO(ChatIO):
    # 实现 prompt_for_input 方法，提示用户输入信息
    def prompt_for_input(self, role: str) -> str:
        return input(f"{role}: ")

    # 实现 prompt_for_output 方法，输出角色的信息并刷新
    def prompt_for_output(self, role: str) -> str:
        print(f"{role}: ", end="", flush=True)

    # 实现 stream_output 方法，流式输出输出流中的内容
    def stream_output(self, output_stream, skip_echo_len: int):
        # 初始化上一个输出长度为 0
        pre = 0
        # 遍历输出流中的每个输出
        for outputs in output_stream:
            # 去除输出中指定长度的前缀，并去除首尾空白字符
            outputs = outputs[skip_echo_len:].strip()
            # 计算当前输出长度
            now = len(outputs) - 1
            # 如果当前长度大于上一个长度
            if now > pre:
                # 输出当前输出中从上一个长度到当前长度的内容，并刷新
                print(" ".join(outputs[pre:now]), end=" ", flush=True)
                # 更新上一个长度为当前长度
                pre = now

        # 输出剩余的内容并刷新
        print(" ".join(outputs[pre:]), flush=True)
        # 返回连接后的输出内容
        return " ".join(outputs)
```