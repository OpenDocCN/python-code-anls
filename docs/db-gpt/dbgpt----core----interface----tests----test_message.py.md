# `.\DB-GPT-src\dbgpt\core\interface\tests\test_message.py`

```py
import pytest  # 导入pytest模块

from dbgpt.core.interface.message import *  # 导入dbgpt.core.interface.message模块下的所有内容
from dbgpt.core.interface.tests.conftest import in_memory_storage  # 导入dbgpt.core.interface.tests.conftest模块中的in_memory_storage函数


@pytest.fixture
def basic_conversation():
    # 返回一个OnceConversation对象，初始化参数为chat_mode="chat_normal", user_name="user1", sys_code="sys1"
    return OnceConversation(chat_mode="chat_normal", user_name="user1", sys_code="sys1")


@pytest.fixture
def human_message():
    # 返回一个HumanMessage对象，内容为"Hello"
    return HumanMessage(content="Hello")


@pytest.fixture
def ai_message():
    # 返回一个AIMessage对象，内容为"Hi there"
    return AIMessage(content="Hi there")


@pytest.fixture
def system_message():
    # 返回一个SystemMessage对象，内容为"System update"
    return SystemMessage(content="System update")


@pytest.fixture
def view_message():
    # 返回一个ViewMessage对象，内容为"View this"
    return ViewMessage(content="View this")


@pytest.fixture
def conversation_identifier():
    # 返回一个ConversationIdentifier对象，标识为"conv1"
    return ConversationIdentifier("conv1")


@pytest.fixture
def message_identifier():
    # 返回一个MessageIdentifier对象，标识为"conv1", 1
    return MessageIdentifier("conv1", 1)


@pytest.fixture
def message_storage_item():
    # 创建一个HumanMessage对象，内容为"Hello"，索引为1，并将其转换为字典格式
    message = HumanMessage(content="Hello", index=1)
    message_detail = message.to_dict()
    # 返回一个MessageStorageItem对象，标识为"conv1", 1，包含消息详细信息
    return MessageStorageItem("conv1", 1, message_detail)


@pytest.fixture
def storage_conversation():
    # 返回一个StorageConversation对象，标识为"conv1"，初始化参数为chat_mode="chat_normal", user_name="user1"
    return StorageConversation("conv1", chat_mode="chat_normal", user_name="user1")


@pytest.fixture
def conversation_with_messages():
    # 创建一个OnceConversation对象，初始化参数为chat_mode="chat_normal", user_name="user1"
    conv = OnceConversation(chat_mode="chat_normal", user_name="user1")
    # 开始一个新的会话轮次
    conv.start_new_round()
    # 添加用户消息"Hello"
    conv.add_user_message("Hello")
    # 添加AI消息"Hi"
    conv.add_ai_message("Hi")
    # 结束当前会话轮次

    conv.end_current_round()
    # 开始一个新的会话轮次
    conv.start_new_round()
    # 添加用户消息"How are you?"
    conv.add_user_message("How are you?")
    # 添加AI消息"I'm good, thanks"
    conv.add_ai_message("I'm good, thanks")
    # 结束当前会话轮次

    conv.end_current_round()
    # 返回包含多个消息的OnceConversation对象
    return conv


@pytest.fixture
def human_model_message():
    # 返回一个ModelMessage对象，角色为ModelMessageRoleType.HUMAN，内容为"Hello"
    return ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hello")


@pytest.fixture
def ai_model_message():
    # 返回一个ModelMessage对象，角色为ModelMessageRoleType.AI，内容为"Hi there"
    return ModelMessage(role=ModelMessageRoleType.AI, content="Hi there")


@pytest.fixture
def system_model_message():
    # 返回一个ModelMessage对象，角色为ModelMessageRoleType.SYSTEM，内容为"You are a helpful chatbot!"
    return ModelMessage(
        role=ModelMessageRoleType.SYSTEM, content="You are a helpful chatbot!"
    )


def test_init(basic_conversation):
    # 断言基本会话对象的初始化参数和默认值
    assert basic_conversation.chat_mode == "chat_normal"
    assert basic_conversation.user_name == "user1"
    assert basic_conversation.sys_code == "sys1"
    assert basic_conversation.messages == []
    assert basic_conversation.start_date == ""
    assert basic_conversation.chat_order == 0
    assert basic_conversation.model_name == ""
    assert basic_conversation.param_type == ""
    assert basic_conversation.param_value == ""
    assert basic_conversation.cost == 0
    assert basic_conversation.tokens == 0
    assert basic_conversation._message_index == 0


def test_add_user_message(basic_conversation, human_message):
    # 添加用户消息到基本会话对象，并断言消息列表长度和消息类型
    basic_conversation.add_user_message(human_message.content)
    assert len(basic_conversation.messages) == 1
    assert isinstance(basic_conversation.messages[0], HumanMessage)


def test_add_ai_message(basic_conversation, ai_message):
    # 添加AI消息到基本会话对象，并断言消息列表长度和消息类型
    basic_conversation.add_ai_message(ai_message.content)
    assert len(basic_conversation.messages) == 1
    assert isinstance(basic_conversation.messages[0], AIMessage)
# 向基本对话中添加系统消息，并验证消息列表长度为1
def test_add_system_message(basic_conversation, system_message):
    basic_conversation.add_system_message(system_message.content)
    assert len(basic_conversation.messages) == 1
    assert isinstance(basic_conversation.messages[0], SystemMessage)


# 向基本对话中添加视图消息，并验证消息列表长度为1
def test_add_view_message(basic_conversation, view_message):
    basic_conversation.add_view_message(view_message.content)
    assert len(basic_conversation.messages) == 1
    assert isinstance(basic_conversation.messages[0], ViewMessage)


# 设置基本对话的起始时间为当前时间，并验证其格式化后的起始日期
def test_set_start_time(basic_conversation):
    now = datetime.now()
    basic_conversation.set_start_time(now)
    assert basic_conversation.start_date == now.strftime("%Y-%m-%d %H:%M:%S")


# 向基本对话中添加用户消息，然后清空消息列表，并验证列表长度为0
def test_clear_messages(basic_conversation, human_message):
    basic_conversation.add_user_message(human_message.content)
    basic_conversation.clear()
    assert len(basic_conversation.messages) == 0


# 向基本对话中添加用户消息，获取最新的用户消息，并验证内容与添加的内容一致
def test_get_latest_user_message(basic_conversation, human_message):
    basic_conversation.add_user_message(human_message.content)
    latest_message = basic_conversation.get_latest_user_message()
    assert latest_message.content == human_message.content


# 向基本对话中添加系统消息，获取所有系统消息，并验证消息列表长度为1，内容与添加的系统消息内容一致
def test_get_system_messages(basic_conversation, system_message):
    basic_conversation.add_system_message(system_message.content)
    system_messages = basic_conversation.get_system_messages()
    assert len(system_messages) == 1
    assert system_messages[0].content == system_message.content


# 从新的对话实例中获取属性值，并验证基本对话的属性已更新
def test_from_conversation(basic_conversation):
    new_conversation = OnceConversation(chat_mode="chat_advanced", user_name="user2")
    basic_conversation.from_conversation(new_conversation)
    assert basic_conversation.chat_mode == "chat_advanced"
    assert basic_conversation.user_name == "user2"


# 获取指定轮次的消息列表，并验证第一轮的消息数量和内容，以及不存在第三轮时返回的空消息列表
def test_get_messages_by_round(conversation_with_messages):
    # Test first round
    round1_messages = conversation_with_messages.get_messages_by_round(1)
    assert len(round1_messages) == 2
    assert round1_messages[0].content == "Hello"
    assert round1_messages[1].content == "Hi"

    # Test not existing round
    no_messages = conversation_with_messages.get_messages_by_round(3)
    assert len(no_messages) == 0


# 获取最新的消息轮次，并验证消息数量和内容与预期一致
def test_get_latest_round(conversation_with_messages):
    latest_round_messages = conversation_with_messages.get_latest_round()
    assert len(latest_round_messages) == 2
    assert latest_round_messages[0].content == "How are you?"
    assert latest_round_messages[1].content == "I'm good, thanks"


# 获取包含指定轮次消息的列表，并验证最后一轮和最后两轮的消息数量和内容
def test_get_messages_with_round(conversation_with_messages):
    # Test last round
    last_round_messages = conversation_with_messages.get_messages_with_round(1)
    assert len(last_round_messages) == 2
    assert last_round_messages[0].content == "How are you?"
    assert last_round_messages[1].content == "I'm good, thanks"

    # Test last two rounds
    last_two_rounds_messages = conversation_with_messages.get_messages_with_round(2)
    assert len(last_two_rounds_messages) == 4
    # 断言第一个消息的内容是否为 "Hello"
    assert last_two_rounds_messages[0].content == "Hello"
    # 断言第二个消息的内容是否为 "Hi"
    assert last_two_rounds_messages[1].content == "Hi"
# 测试获取对话中的模型消息
def test_get_model_messages(conversation_with_messages):
    # 调用对话对象的方法获取模型消息
    model_messages = conversation_with_messages.get_model_messages()
    # 断言模型消息的数量为4
    assert len(model_messages) == 4
    # 断言所有模型消息都是 ModelMessage 类型
    assert all(isinstance(msg, ModelMessage) for msg in model_messages)
    # 断言第一个模型消息的内容为"Hello"
    assert model_messages[0].content == "Hello"
    # 断言第二个模型消息的内容为"Hi"
    assert model_messages[1].content == "Hi"
    # 断言第三个模型消息的内容为"How are you?"
    assert model_messages[2].content == "How are you?"
    # 断言第四个模型消息的内容为"I'm good, thanks"
    assert model_messages[3].content == "I'm good, thanks"


# 测试对话标识符
def test_conversation_identifier(conversation_identifier):
    # 断言对话标识符的 conv_uid 为"conv1"
    assert conversation_identifier.conv_uid == "conv1"
    # 断言对话标识符的 identifier_type 为"conversation"
    assert conversation_identifier.identifier_type == "conversation"
    # 断言对话标识符的 str_identifier 为"conversation:conv1"
    assert conversation_identifier.str_identifier == "conversation:conv1"
    # 断言对话标识符转换为字典的结果符合预期
    assert conversation_identifier.to_dict() == {
        "conv_uid": "conv1",
        "identifier_type": "conversation",
    }


# 测试消息标识符
def test_message_identifier(message_identifier):
    # 断言消息标识符的 conv_uid 为"conv1"
    assert message_identifier.conv_uid == "conv1"
    # 断言消息标识符的 index 为1
    assert message_identifier.index == 1
    # 断言消息标识符的 identifier_type 为"message"
    assert message_identifier.identifier_type == "message"
    # 断言消息标识符的 str_identifier 为"message___conv1___1"
    assert message_identifier.str_identifier == "message___conv1___1"
    # 断言消息标识符转换为字典的结果符合预期
    assert message_identifier.to_dict() == {
        "conv_uid": "conv1",
        "index": 1,
        "identifier_type": "message",
    }


# 测试消息存储项
def test_message_storage_item(message_storage_item):
    # 断言消息存储项的 conv_uid 为"conv1"
    assert message_storage_item.conv_uid == "conv1"
    # 断言消息存储项的 index 为1
    assert message_storage_item.index == 1
    # 断言消息存储项的 message_detail 符合预期
    assert message_storage_item.message_detail == {
        "type": "human",
        "data": {
            "content": "Hello",
            "index": 1,
            "round_index": 0,
            "additional_kwargs": {},
            "example": False,
        },
        "index": 1,
        "round_index": 0,
    }

    # 断言消息存储项的 identifier 是 MessageIdentifier 类型
    assert isinstance(message_storage_item.identifier, MessageIdentifier)
    # 断言消息存储项转换为字典的结果符合预期
    assert message_storage_item.to_dict() == {
        "conv_uid": "conv1",
        "index": 1,
        "message_detail": {
            "type": "human",
            "index": 1,
            "data": {
                "content": "Hello",
                "index": 1,
                "round_index": 0,
                "additional_kwargs": {},
                "example": False,
            },
            "round_index": 0,
        },
    }

    # 断言消息存储项转换为消息对象的结果是 BaseMessage 类型
    assert isinstance(message_storage_item.to_message(), BaseMessage)


# 测试存储对话初始化
def test_storage_conversation_init(storage_conversation):
    # 断言存储对话的 conv_uid 为"conv1"
    assert storage_conversation.conv_uid == "conv1"
    # 断言存储对话的 chat_mode 为"chat_normal"
    assert storage_conversation.chat_mode == "chat_normal"
    # 断言存储对话的 user_name 为"user1"


# 测试存储对话添加用户消息
def test_storage_conversation_add_user_message(storage_conversation):
    # 添加用户消息"Hi"
    storage_conversation.add_user_message("Hi")
    # 断言存储对话的消息数量为1
    assert len(storage_conversation.messages) == 1
    # 断言存储对话的第一个消息是 HumanMessage 类型
    assert isinstance(storage_conversation.messages[0], HumanMessage)


# 测试存储对话添加 AI 消息
def test_storage_conversation_add_ai_message(storage_conversation):
    # 添加 AI 消息"Hello"
    storage_conversation.add_ai_message("Hello")
    # 断言存储对话的消息数量为1
    assert len(storage_conversation.messages) == 1
    # 断言存储对话的第一个消息是 AIMessage 类型
    assert isinstance(storage_conversation.messages[0], AIMessage)
# 定义一个测试函数，用于测试存储对话到存储系统的功能
def test_save_to_storage(storage_conversation, in_memory_storage):
    # 设置存储对话和消息存储为内存存储
    storage_conversation.conv_storage = in_memory_storage
    storage_conversation.message_storage = in_memory_storage

    # 添加用户消息和AI回复消息到对话
    storage_conversation.add_user_message("User message")
    storage_conversation.add_ai_message("AI response")

    # 将对话保存到存储系统中
    storage_conversation.save_to_storage()

    # 创建一个新的StorageConversation实例来加载数据
    saved_conversation = StorageConversation(
        storage_conversation.conv_uid,
        conv_storage=in_memory_storage,
        message_storage=in_memory_storage,
    )

    # 断言确保保存的数据正确加载
    assert saved_conversation.conv_uid == storage_conversation.conv_uid
    assert len(saved_conversation.messages) == 2
    assert isinstance(saved_conversation.messages[0], HumanMessage)
    assert isinstance(saved_conversation.messages[1], AIMessage)


# 定义一个测试函数，用于测试从存储系统加载对话的功能
def test_load_from_storage(storage_conversation, in_memory_storage):
    # 设置存储对话和消息存储为内存存储
    storage_conversation.conv_storage = in_memory_storage
    storage_conversation.message_storage = in_memory_storage

    # 添加用户消息和AI回复消息到对话，并保存到存储系统中
    storage_conversation.add_user_message("User message")
    storage_conversation.add_ai_message("AI response")
    storage_conversation.save_to_storage()

    # 创建一个新的StorageConversation实例来加载数据
    new_conversation = StorageConversation(
        "conv1", conv_storage=in_memory_storage, message_storage=in_memory_storage
    )

    # 检查加载的数据是否正确
    assert new_conversation.conv_uid == storage_conversation.conv_uid
    assert len(new_conversation.messages) == 2
    assert new_conversation.messages[0].content == "User message"
    assert new_conversation.messages[1].content == "AI response"
    assert isinstance(new_conversation.messages[0], HumanMessage)
    assert isinstance(new_conversation.messages[1], AIMessage)


# 定义一个测试函数，用于测试从存储系统删除对话的功能
def test_delete(storage_conversation, in_memory_storage):
    # 设置存储对话和消息存储为内存存储
    storage_conversation.conv_storage = in_memory_storage
    storage_conversation.message_storage = in_memory_storage

    # 开始一个新的对话轮次，添加用户消息和AI回复消息，并结束当前轮次
    storage_conversation.start_new_round()
    storage_conversation.add_user_message("User message")
    storage_conversation.add_ai_message("AI response")
    storage_conversation.end_current_round()

    # 创建一个新的StorageConversation实例来加载数据
    new_conversation = StorageConversation(
        "conv1", conv_storage=in_memory_storage, message_storage=in_memory_storage
    )

    # 删除对话
    new_conversation.delete()

    # 检查对话是否被成功删除
    assert new_conversation.conv_uid == storage_conversation.conv_uid
    assert len(new_conversation.messages) == 0

    # 创建一个新的StorageConversation实例来加载没有消息的对话
    no_messages_conv = StorageConversation(
        "conv1", conv_storage=in_memory_storage, message_storage=in_memory_storage
    )
    assert len(no_messages_conv.messages) == 0
# 定义一个测试函数，用于测试解析模型消息时没有历史消息的情况
def test_parse_model_messages_no_history_messages():
    # 创建包含单条消息的列表，消息为人类角色，内容为 "Hello"
    messages = [
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hello"),
    ]
    # 调用解析模型消息函数，返回用户提示、系统消息和历史消息
    user_prompt, system_messages, history_messages = parse_model_messages(messages)
    # 断言用户提示应为 "Hello"
    assert user_prompt == "Hello"
    # 断言系统消息应为空列表
    assert system_messages == []
    # 断言历史消息应为空列表
    assert history_messages == []

# 定义一个测试函数，用于测试解析单轮会话的情况
def test_parse_model_messages_single_round_conversation():
    # 创建包含三条消息的列表，依次为人类角色 "Hello"、AI角色 "Hi there!"、人类角色 "Hello again"
    messages = [
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hello"),
        ModelMessage(role=ModelMessageRoleType.AI, content="Hi there!"),
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hello again"),
    ]
    # 调用解析模型消息函数，返回用户提示、系统消息和历史消息
    user_prompt, system_messages, history_messages = parse_model_messages(messages)
    # 断言用户提示应为 "Hello again"
    assert user_prompt == "Hello again"
    # 断言系统消息应为空列表
    assert system_messages == []
    # 断言历史消息应为包含两条消息的列表 [["Hello", "Hi there!"]]
    assert history_messages == [["Hello", "Hi there!"]]

# 定义一个测试函数，用于测试解析包含系统消息的两轮会话的情况
def test_parse_model_messages_two_round_conversation_with_system_message():
    # 创建包含四条消息的列表，依次为系统角色 "System initializing..."、人类角色 "How's the weather?"、
    # AI角色 "It's sunny!"、人类角色 "Great to hear!"
    messages = [
        ModelMessage(role=ModelMessageRoleType.SYSTEM, content="System initializing..."),
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="How's the weather?"),
        ModelMessage(role=ModelMessageRoleType.AI, content="It's sunny!"),
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="Great to hear!"),
    ]
    # 调用解析模型消息函数，返回用户提示、系统消息和历史消息
    user_prompt, system_messages, history_messages = parse_model_messages(messages)
    # 断言用户提示应为 "Great to hear!"
    assert user_prompt == "Great to hear!"
    # 断言系统消息应为包含一条消息的列表 ["System initializing..."]
    assert system_messages == ["System initializing..."]
    # 断言历史消息应为包含两条消息的列表 [["How's the weather?", "It's sunny!"]]
    assert history_messages == [["How's the weather?", "It's sunny!"]]

# 定义一个测试函数，用于测试解析三轮会话的情况
def test_parse_model_messages_three_round_conversation():
    # 创建包含五条消息的列表，依次为人类角色 "Hi"、AI角色 "Hello!"、人类角色 "What's up?"、
    # AI角色 "Not much, you?"、人类角色 "Same here."
    messages = [
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hi"),
        ModelMessage(role=ModelMessageRoleType.AI, content="Hello!"),
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="What's up?"),
        ModelMessage(role=ModelMessageRoleType.AI, content="Not much, you?"),
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="Same here."),
    ]
    # 调用解析模型消息函数，返回用户提示、系统消息和历史消息
    user_prompt, system_messages, history_messages = parse_model_messages(messages)
    # 断言用户提示应为 "Same here."
    assert user_prompt == "Same here."
    # 断言系统消息应为空列表
    assert system_messages == []
    # 断言历史消息应为包含两条消息的列表 [["Hi", "Hello!"], ["What's up?", "Not much, you?"]]
    assert history_messages == [["Hi", "Hello!"], ["What's up?", "Not much, you?"]]

# 定义一个测试函数，用于测试解析包含多条系统消息的情况
def test_parse_model_messages_multiple_system_messages():
    # 创建包含五条消息的列表，依次为系统角色 "System start"、人类角色 "Hey"、AI角色 "Hello!"、
    # 系统角色 "System check"、人类角色 "How are you?"
    messages = [
        ModelMessage(role=ModelMessageRoleType.SYSTEM, content="System start"),
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="Hey"),
        ModelMessage(role=ModelMessageRoleType.AI, content="Hello!"),
        ModelMessage(role=ModelMessageRoleType.SYSTEM, content="System check"),
        ModelMessage(role=ModelMessageRoleType.HUMAN, content="How are you?"),
    ]
    # 调用解析模型消息函数，返回用户提示、系统消息和历史消息
    user_prompt, system_messages, history_messages = parse_model_messages(messages)
    # 断言用户提示应为 "How are you?"
    assert user_prompt == "How are you?"
    # 断言系统消息应为包含两条消息的列表 ["System start", "System check"]
    assert system_messages == ["System start", "System check"]
    # 断言历史消息应为包含一条消息的列表 [["Hey", "Hello!"]]
    assert history_messages == [["Hey", "Hello!"]]
    # 分别定义三个变量，用于存储人类模型的消息、AI模型的消息和系统模型的消息
    human_model_message, ai_model_message, system_model_message
# 将空列表转换为通用消息格式，预期结果是空列表
none_messages = ModelMessage.to_common_messages([])
assert none_messages == []

# 将单个人类模型消息转换为通用消息格式，包含用户角色和消息内容
single_messages = ModelMessage.to_common_messages([human_model_message])
assert single_messages == [{"role": "user", "content": human_model_message.content}]

# 将多个模型消息转换为通用消息格式，保留每个消息的角色和内容
normal_messages = ModelMessage.to_common_messages(
    [
        system_model_message,
        human_model_message,
        ai_model_message,
        human_model_message,
    ]
)
assert normal_messages == [
    {"role": "system", "content": system_model_message.content},
    {"role": "user", "content": human_model_message.content},
    {"role": "assistant", "content": ai_model_message.content},
    {"role": "user", "content": human_model_message.content},
]

# 测试将消息转换为兼容的 OpenAI 消息格式
def test_to_openai_messages_convert_to_compatible_format(
    human_model_message, ai_model_message, system_model_message
):
    # 混洗并将消息转换为兼容的 OpenAI 消息格式，保留每个消息的角色和内容
    shuffle_messages = ModelMessage.to_common_messages(
        [
            system_model_message,
            human_model_message,
            human_model_message,
            ai_model_message,
        ],
        convert_to_compatible_format=True,
    )
    assert shuffle_messages == [
        {"role": "system", "content": system_model_message.content},
        {"role": "user", "content": human_model_message.content},
        {"role": "assistant", "content": ai_model_message.content},
        {"role": "user", "content": human_model_message.content},
    ]
```