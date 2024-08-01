# `.\DB-GPT-src\dbgpt\core\interface\operators\tests\test_message_operator.py`

```py
# 导入必要的模块和类
from typing import List
import pytest
from dbgpt.core.interface.message import AIMessage, BaseMessage, HumanMessage
from dbgpt.core.operators import BufferedConversationMapperOperator

# 定义测试用例前置条件 fixture，返回包含多条消息的列表
@pytest.fixture
def messages() -> List[BaseMessage]:
    return [
        HumanMessage(content="Hi", round_index=1),       # 创建人类消息对象，内容为 "Hi"，轮次为 1
        AIMessage(content="Hello!", round_index=1),      # 创建 AI 消息对象，内容为 "Hello!"，轮次为 1
        HumanMessage(content="How are you?", round_index=2),  # 创建人类消息对象，内容为 "How are you?"，轮次为 2
        AIMessage(content="I'm good, thanks!", round_index=2),  # 创建 AI 消息对象，内容为 "I'm good, thanks!"，轮次为 2
        HumanMessage(content="What's new today?", round_index=3),  # 创建人类消息对象，内容为 "What's new today?"，轮次为 3
        AIMessage(content="Lots of things!", round_index=3),  # 创建 AI 消息对象，内容为 "Lots of things!"，轮次为 3
    ]

# 定义异步测试函数，测试保留起始轮次的操作
@pytest.mark.asyncio
async def test_buffered_conversation_keep_start_rounds(messages: List[BaseMessage]):
    # 测试 keep_start_rounds 参数为 2
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=2,     # 初始化 BufferedConversationMapperOperator 实例，保留起始 2 轮次的消息
        keep_end_rounds=None,    # keep_end_rounds 参数为 None，即不保留结束轮次的消息
    )
    assert await operator.map_messages(messages) == [
        HumanMessage(content="Hi", round_index=1),
        AIMessage(content="Hello!", round_index=1),
        HumanMessage(content="How are you?", round_index=2),
        AIMessage(content="I'm good, thanks!", round_index=2),
    ]

    # 测试 keep_start_rounds 参数为 0
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=0,     # 初始化 BufferedConversationMapperOperator 实例，不保留起始轮次的消息
        keep_end_rounds=None,
    )
    assert await operator.map_messages(messages) == []  # 预期结果为空列表

    # 测试 keep_start_rounds 参数为 100，超过实际消息轮次
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=100,   # 初始化 BufferedConversationMapperOperator 实例，保留起始 100 轮次的消息
        keep_end_rounds=None,
    )
    assert await operator.map_messages(messages) == messages  # 预期结果为完整的消息列表

    # 测试 keep_start_rounds 参数为负数，应引发 ValueError 异常
    with pytest.raises(ValueError):
        operator = BufferedConversationMapperOperator(
            keep_start_rounds=-1,  # 初始化 BufferedConversationMapperOperator 实例，传入负数参数
            keep_end_rounds=None,
        )
        await operator.map_messages(messages)  # 应该抛出异常

# 定义异步测试函数，测试保留结束轮次的操作
@pytest.mark.asyncio
async def test_buffered_conversation_keep_end_rounds(messages: List[BaseMessage]):
    # 测试 keep_end_rounds 参数为 2
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=None,  # keep_start_rounds 参数为 None，即不保留起始轮次的消息
        keep_end_rounds=2,       # 初始化 BufferedConversationMapperOperator 实例，保留结束 2 轮次的消息
    )
    assert await operator.map_messages(messages) == [
        HumanMessage(content="How are you?", round_index=2),
        AIMessage(content="I'm good, thanks!", round_index=2),
        HumanMessage(content="What's new today?", round_index=3),
        AIMessage(content="Lots of things!", round_index=3),
    ]

    # 测试 keep_start_rounds 和 keep_end_rounds 参数均为 0
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=0,
        keep_end_rounds=0,
    )
    assert await operator.map_messages(messages) == []  # 预期结果为空列表

    # 测试 keep_end_rounds 参数为 100，超过实际消息轮次
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=None,
        keep_end_rounds=100,
    )
    assert await operator.map_messages(messages) == messages  # 预期结果为完整的消息列表

    # 测试 keep_end_rounds 参数为负数，应引发 ValueError 异常
    # 此部分代码在原始内容中未提供，可能需要补充完整后再进行注释
    with pytest.raises(ValueError):
        operator = BufferedConversationMapperOperator(
            keep_start_rounds=None,
            keep_end_rounds=-1,  # 初始化 BufferedConversationMapperOperator 实例，传入负数参数
        )
        await operator.map_messages(messages)  # 应该抛出异常
    # 使用 pytest 来断言抛出 ValueError 异常
    with pytest.raises(ValueError):
        # 创建 BufferedConversationMapperOperator 对象，设置参数 keep_start_rounds 为 None，
        # keep_end_rounds 为 -1
        operator = BufferedConversationMapperOperator(
            keep_start_rounds=None,
            keep_end_rounds=-1,
        )
        # 调用 operator 的 map_messages 方法处理 messages
        await operator.map_messages(messages)
# 使用 pytest 框架标记此函数为异步测试函数
@pytest.mark.asyncio
async def test_buffered_conversation_keep_start_end_rounds(messages: List[BaseMessage]):
    # 测试 keep_start_rounds 和 keep_end_rounds 的功能
    
    # 创建 BufferedConversationMapperOperator 对象，保留对话开始和结束各1轮
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=1,
        keep_end_rounds=1,
    )
    # 断言调用 map_messages 方法后返回的结果是否符合预期
    assert await operator.map_messages(messages) == [
        HumanMessage(content="Hi", round_index=1),
        AIMessage(content="Hello!", round_index=1),
        HumanMessage(content="What's new today?", round_index=3),
        AIMessage(content="Lots of things!", round_index=3),
    ]
    
    # 创建 BufferedConversationMapperOperator 对象，不保留任何对话开始和结束轮数
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=0,
        keep_end_rounds=0,
    )
    # 断言调用 map_messages 方法后返回的结果是否符合预期（空列表）
    assert await operator.map_messages(messages) == []

    # 创建 BufferedConversationMapperOperator 对象，不保留对话开始轮数，但保留结束轮数1轮
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=0,
        keep_end_rounds=1,
    )
    # 断言调用 map_messages 方法后返回的结果是否符合预期
    assert await operator.map_messages(messages) == [
        HumanMessage(content="What's new today?", round_index=3),
        AIMessage(content="Lots of things!", round_index=3),
    ]

    # 创建 BufferedConversationMapperOperator 对象，保留对话开始2轮，但不保留任何结束轮数
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=2,
        keep_end_rounds=0,
    )
    # 断言调用 map_messages 方法后返回的结果是否符合预期
    assert await operator.map_messages(messages) == [
        HumanMessage(content="Hi", round_index=1),
        AIMessage(content="Hello!", round_index=1),
        HumanMessage(content="How are you?", round_index=2),
        AIMessage(content="I'm good, thanks!", round_index=2),
    ]

    # 创建 BufferedConversationMapperOperator 对象，保留对话开始和结束各100轮
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=100,
        keep_end_rounds=100,
    )
    # 断言调用 map_messages 方法后返回的结果是否符合预期（与输入消息列表相同）
    assert await operator.map_messages(messages) == messages

    # 创建 BufferedConversationMapperOperator 对象，保留对话开始和结束各2轮
    operator = BufferedConversationMapperOperator(
        keep_start_rounds=2,
        keep_end_rounds=2,
    )
    # 断言调用 map_messages 方法后返回的结果是否符合预期（与输入消息列表相同）
    assert await operator.map_messages(messages) == messages

    # 测试保留负数轮数时是否会引发 ValueError 异常
    with pytest.raises(ValueError):
        operator = BufferedConversationMapperOperator(
            keep_start_rounds=-1,
            keep_end_rounds=-1,
        )
        # 调用 map_messages 方法，预期会抛出 ValueError 异常
        await operator.map_messages(messages)
```