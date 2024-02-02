# `MetaGPT\tests\mock\mock_llm.py`

```py

# 从 typing 模块中导入 Optional 类型
from typing import Optional

# 从 metagpt.logs 模块中导入 log_llm_stream 和 logger
from metagpt.logs import log_llm_stream, logger
# 从 metagpt.provider.openai_api 模块中导入 OpenAILLM 类
from metagpt.provider.openai_api import OpenAILLM

# 定义 MockLLM 类，继承自 OpenAILLM 类
class MockLLM(OpenAILLM):
    # 初始化方法，接受 allow_open_api_call 参数
    def __init__(self, allow_open_api_call):
        # 调用父类的初始化方法
        super().__init__()
        # 设置 allow_open_api_call 属性
        self.allow_open_api_call = allow_open_api_call
        # 初始化 rsp_cache 属性为字典
        self.rsp_cache: dict = {}
        # 初始化 rsp_candidates 属性为列表，用于存储多次调用的结果
        self.rsp_candidates: list[dict] = []  # a test can have multiple calls with the same llm, thus a list

    # 异步方法，接受 messages、stream、timeout 参数，返回字符串
    async def acompletion_text(self, messages: list[dict], stream=False, timeout=3) -> str:
        """Overwrite original acompletion_text to cancel retry"""
        # 如果 stream 为 True
        if stream:
            # 调用 _achat_completion_stream 方法，传入 messages 和 timeout 参数
            resp = self._achat_completion_stream(messages, timeout=timeout)

            # 初始化 collected_messages 列表
            collected_messages = []
            # 异步迭代 resp
            async for i in resp:
                # 记录日志
                log_llm_stream(i)
                # 将消息添加到 collected_messages 中
                collected_messages.append(i)

            # 将 collected_messages 中的消息连接成字符串
            full_reply_content = "".join(collected_messages)
            # 计算使用情况
            usage = self._calc_usage(messages, full_reply_content)
            # 更新成本
            self._update_costs(usage)
            # 返回完整的回复内容
            return full_reply_content

        # 否则
        rsp = await self._achat_completion(messages, timeout=timeout)
        # 返回选择的文本
        return self.get_choice_text(rsp)

    # 异步方法，接受 msg、system_msgs、format_msgs、timeout、stream 参数，返回字符串
    async def original_aask(
        self,
        msg: str,
        system_msgs: Optional[list[str]] = None,
        format_msgs: Optional[list[dict[str, str]]] = None,
        timeout=3,
        stream=True,
    ):
        """A copy of metagpt.provider.base_llm.BaseLLM.aask, we can't use super().aask because it will be mocked"""
        # 如果 system_msgs 存在
        if system_msgs:
            # 生成系统消息
            message = self._system_msgs(system_msgs)
        else:
            # 否则，使用默认系统消息
            message = [self._default_system_msg()] if self.use_system_prompt else []
        # 如果 format_msgs 存在
        if format_msgs:
            # 将 format_msgs 添加到消息中
            message.extend(format_msgs)
        # 将用户消息添加到消息中
        message.append(self._user_msg(msg))
        # 调用 acompletion_text 方法，传入消息、stream 和 timeout 参数
        rsp = await self.acompletion_text(message, stream=stream, timeout=timeout)
        # 返回响应
        return rsp

    # 异步方法，接受 msgs、timeout 参数，返回字符串
    async def original_aask_batch(self, msgs: list, timeout=3) -> str:
        """A copy of metagpt.provider.base_llm.BaseLLM.aask_batch, we can't use super().aask because it will be mocked"""
        # 初始化上下文
        context = []
        # 遍历消息列表
        for msg in msgs:
            # 将用户消息添加到上下文中
            umsg = self._user_msg(msg)
            context.append(umsg)
            # 调用 acompletion_text 方法，传入上下文和 timeout 参数
            rsp_text = await self.acompletion_text(context, timeout=timeout)
            # 将助手消息添加到上下文中
            context.append(self._assistant_msg(rsp_text))
        # 提取助手响应
        return self._extract_assistant_rsp(context)

    # 异步方法，接受 msg、system_msgs、format_msgs、timeout、stream 参数，返回字符串
    async def aask(
        self,
        msg: str,
        system_msgs: Optional[list[str]] = None,
        format_msgs: Optional[list[dict[str, str]]] = None,
        timeout=3,
        stream=True,
    ) -> str:
        # 用于标识消息是否已经被调用过
        msg_key = msg
        # 如果 system_msgs 存在
        if system_msgs:
            # 将系统消息连接成字符串
            joined_system_msg = "#MSG_SEP#".join(system_msgs) + "#SYSTEM_MSG_END#"
            msg_key = joined_system_msg + msg_key
        # 调用 _mock_rsp 方法，传入消息键和其他参数
        rsp = await self._mock_rsp(msg_key, self.original_aask, msg, system_msgs, format_msgs, timeout, stream)
        # 返回响应
        return rsp

    # 异步方法，接受 msgs、timeout 参数，返回字符串
    async def aask_batch(self, msgs: list, timeout=3) -> str:
        # 将消息列表连接成字符串
        msg_key = "#MSG_SEP#".join([msg if isinstance(msg, str) else msg.content for msg in msgs])
        # 调用 _mock_rsp 方法，传入消息键和其他参数
        rsp = await self._mock_rsp(msg_key, self.original_aask_batch, msgs, timeout)
        # 返回响应
        return rsp

    # 异步方法，接受消息键、ask_func 和其他参数，返回响应
    async def _mock_rsp(self, msg_key, ask_func, *args, **kwargs):
        # 如果消息键不在缓存中
        if msg_key not in self.rsp_cache:
            # 如果不允许调用 OpenAI API
            if not self.allow_open_api_call:
                # 抛出异常
                raise ValueError(
                    "In current test setting, api call is not allowed, you should properly mock your tests, "
                    "or add expected api response in tests/data/rsp_cache.json. "
                    f"The prompt you want for api call: {msg_key}"
                )
            # 否则，调用原始未模拟的方法
            rsp = await ask_func(*args, **kwargs)
        else:
            # 否则，使用响应缓存
            logger.warning("Use response cache")
            rsp = self.rsp_cache[msg_key]
        # 将响应添加到候选响应列表中
        self.rsp_candidates.append({msg_key: rsp})
        # 返回响应
        return rsp

```