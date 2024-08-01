# `.\DB-GPT-src\dbgpt\storage\chat_history\tests\test_storage_adapter.py`

```py
# 引入 List 类型，用于声明列表类型的变量或返回值
from typing import List

# 引入 pytest 模块，用于编写和运行测试用例
import pytest

# 从 dbgpt.core.interface.message 模块中引入 AIMessage, HumanMessage, StorageConversation 类
from dbgpt.core.interface.message import AIMessage, HumanMessage, StorageConversation

# 从 dbgpt.core.interface.storage 模块中引入 QuerySpec 类
from dbgpt.core.interface.storage import QuerySpec

# 从 dbgpt.storage.chat_history.chat_history_db 模块中引入 ChatHistoryEntity, ChatHistoryMessageEntity 类
from dbgpt.storage.chat_history.chat_history_db import (
    ChatHistoryEntity,
    ChatHistoryMessageEntity,
)

# 从 dbgpt.storage.chat_history.storage_adapter 模块中引入 DBMessageStorageItemAdapter, DBStorageConversationItemAdapter 类
from dbgpt.storage.chat_history.storage_adapter import (
    DBMessageStorageItemAdapter,
    DBStorageConversationItemAdapter,
)

# 从 dbgpt.storage.metadata 模块中引入 db 对象
from dbgpt.storage.metadata import db

# 从 dbgpt.storage.metadata.db_storage 模块中引入 SQLAlchemyStorage 类
from dbgpt.storage.metadata.db_storage import SQLAlchemyStorage

# 从 dbgpt.util.pagination_utils 模块中引入 PaginationResult 类
from dbgpt.util.pagination_utils import PaginationResult

# 从 dbgpt.util.serialization.json_serialization 模块中引入 JsonSerializer 类
from dbgpt.util.serialization.json_serialization import JsonSerializer


# 定义一个 fixture 函数，返回一个 JsonSerializer 实例用于测试
@pytest.fixture
def serializer():
    return JsonSerializer()


# 定义一个 fixture 函数，返回一个内存中的 SQLite 数据库 URL 用于测试
@pytest.fixture
def db_url():
    """Use in-memory SQLite database for testing"""
    return "sqlite:///:memory:"
    # return "sqlite:///test.db"


# 定义一个 fixture 函数，初始化并创建测试数据库，返回 db 对象
@pytest.fixture
def db_manager(db_url):
    db.init_db(db_url)
    db.create_all()
    return db


# 定义一个 fixture 函数，返回一个 DBStorageConversationItemAdapter 实例用于测试
@pytest.fixture
def storage_adapter():
    return DBStorageConversationItemAdapter()


# 定义一个 fixture 函数，返回一个 DBMessageStorageItemAdapter 实例用于测试
@pytest.fixture
def storage_message_adapter():
    return DBMessageStorageItemAdapter()


# 定义一个 fixture 函数，返回一个 SQLAlchemyStorage 实例用于测试对话存储
@pytest.fixture
def conv_storage(db_manager, serializer, storage_adapter):
    storage = SQLAlchemyStorage(
        db_manager,
        ChatHistoryEntity,
        storage_adapter,
        serializer,
    )
    return storage


# 定义一个 fixture 函数，返回一个 SQLAlchemyStorage 实例用于测试消息存储
@pytest.fixture
def message_storage(db_manager, serializer, storage_message_adapter):
    storage = SQLAlchemyStorage(
        db_manager,
        ChatHistoryMessageEntity,
        storage_message_adapter,
        serializer,
    )
    return storage


# 定义一个 fixture 函数，返回一个 StorageConversation 实例用于测试会话
@pytest.fixture
def conversation(conv_storage, message_storage):
    return StorageConversation(
        "conv1",
        chat_mode="chat_normal",
        user_name="user1",
        conv_storage=conv_storage,
        message_storage=message_storage,
    )


# 定义一个 fixture 函数，返回一个已经进行四轮对话的 StorageConversation 实例用于测试
@pytest.fixture
def four_round_conversation(conv_storage, message_storage):
    conversation = StorageConversation(
        "conv1",
        chat_mode="chat_normal",
        user_name="user1",
        conv_storage=conv_storage,
        message_storage=message_storage,
    )
    conversation.start_new_round()
    conversation.add_user_message("hello, this is first round")
    conversation.add_ai_message("hi")
    conversation.end_current_round()
    conversation.start_new_round()
    conversation.add_user_message("hello, this is second round")
    conversation.add_ai_message("hi")
    conversation.end_current_round()
    conversation.start_new_round()
    conversation.add_user_message("hello, this is third round")
    conversation.add_ai_message("hi")
    conversation.end_current_round()
    conversation.start_new_round()
    conversation.add_user_message("hello, this is fourth round")
    conversation.add_ai_message("hi")
    conversation.end_current_round()
    return conversation


# 定义一个 fixture 函数，返回一个用于测试的对话列表（未完整）
@pytest.fixture
def conversation_list(request, conv_storage, message_storage):
    # 检查请求对象中是否存在参数对象 request.param，如果存在则使用该参数，否则使用空字典
    params = request.param if hasattr(request, "param") else {}
    # 从参数字典中获取对话轮数参数 conv_count，默认为 4
    conv_count = params.get("conv_count", 4)
    # 初始化空列表 result 用于存储对话对象
    result = []
    # 循环生成指定数量的对话轮数
    for i in range(conv_count):
        # 创建一个新的对话对象 StorageConversation，命名为 conv{i}，设置聊天模式为 chat_normal，用户名为 "user1"
        # 并传入对话存储对象 conv_storage 和消息存储对象 message_storage
        conversation = StorageConversation(
            f"conv{i}",
            chat_mode="chat_normal",
            user_name="user1",
            conv_storage=conv_storage,
            message_storage=message_storage,
        )
        # 开始新的对话轮
        conversation.start_new_round()
        # 添加用户消息到当前对话轮
        conversation.add_user_message("hello, this is first round")
        # 添加AI消息到当前对话轮
        conversation.add_ai_message("hi")
        # 结束当前对话轮
        conversation.end_current_round()
        # 重复以上步骤，创建并处理第二、第三、第四轮对话
        conversation.start_new_round()
        conversation.add_user_message("hello, this is second round")
        conversation.add_ai_message("hi")
        conversation.end_current_round()
        conversation.start_new_round()
        conversation.add_user_message("hello, this is third round")
        conversation.add_ai_message("hi")
        conversation.end_current_round()
        conversation.start_new_round()
        conversation.add_user_message("hello, this is fourth round")
        conversation.add_ai_message("hi")
        conversation.end_current_round()
        # 将处理完的对话对象添加到结果列表中
        result.append(conversation)
    # 返回包含所有对话对象的结果列表
    return result
# 定义一个测试函数，用于测试保存和加载对话内容的功能
def test_save_and_load(
    conversation: StorageConversation, conv_storage, message_storage
):
    # 开始一个新的对话回合
    conversation.start_new_round()
    # 添加用户消息 "hello"
    conversation.add_user_message("hello")
    # 添加AI消息 "hi"
    conversation.add_ai_message("hi")
    # 结束当前对话回合
    conversation.end_current_round()

    # 创建一个保存的对话对象，用于验证保存和加载功能
    saved_conversation = StorageConversation(
        conv_uid=conversation.conv_uid,
        conv_storage=conv_storage,
        message_storage=message_storage,
    )
    # 断言保存的对话UID与原始对话UID相同
    assert saved_conversation.conv_uid == conversation.conv_uid
    # 断言保存的对话消息长度为2
    assert len(saved_conversation.messages) == 2
    # 断言第一条保存的消息是用户消息类型
    assert isinstance(saved_conversation.messages[0], HumanMessage)
    # 断言第二条保存的消息是AI消息类型
    assert isinstance(saved_conversation.messages[1], AIMessage)
    # 断言第一条保存的消息内容为 "hello"
    assert saved_conversation.messages[0].content == "hello"
    # 断言第一条保存的消息回合索引为1
    assert saved_conversation.messages[0].round_index == 1
    # 断言第二条保存的消息内容为 "hi"
    assert saved_conversation.messages[1].content == "hi"
    # 断言第二条保存的消息回合索引为1


# 定义一个测试函数，用于测试查询对话消息的功能
def test_query_message(
    conversation: StorageConversation, conv_storage, message_storage
):
    # 开始一个新的对话回合
    conversation.start_new_round()
    # 添加用户消息 "hello"
    conversation.add_user_message("hello")
    # 添加AI消息 "hi"
    conversation.add_ai_message("hi")
    # 结束当前对话回合
    conversation.end_current_round()

    # 创建一个保存的对话对象，用于验证查询功能
    saved_conversation = StorageConversation(
        conv_uid=conversation.conv_uid,
        conv_storage=conv_storage,
        message_storage=message_storage,
    )
    # 断言保存的对话UID与原始对话UID相同
    assert saved_conversation.conv_uid == conversation.conv_uid
    # 断言保存的对话消息长度为2
    assert len(saved_conversation.messages) == 2

    # 定义查询条件为对话UID的查询规格
    query_spec = QuerySpec(conditions={"conv_uid": conversation.conv_uid})
    # 执行查询，并获取查询结果
    results = conversation.conv_storage.query(query_spec, StorageConversation)
    # 断言查询结果长度为1
    assert len(results) == 1


# 定义一个测试函数，用于测试复杂查询对话的功能
def test_complex_query(
    conversation_list: List[StorageConversation], conv_storage, message_storage
):
    # 定义查询条件为用户名为 "user1" 的查询规格
    query_spec = QuerySpec(conditions={"user_name": "user1"})
    # 执行查询，并获取查询结果
    results = conv_storage.query(query_spec, StorageConversation)
    # 断言查询结果长度与对话列表长度相同
    assert len(results) == len(conversation_list)
    # 遍历查询结果
    for i, result in enumerate(results):
        # 断言每个结果的用户名为 "user1"
        assert result.user_name == "user1"
        # 断言每个结果的对话UID为 "conv{i}"
        assert result.conv_uid == f"conv{i}"
        # 创建一个保存的对话对象，用于验证加载功能
        saved_conversation = StorageConversation(
            conv_uid=result.conv_uid,
            conv_storage=conv_storage,
            message_storage=message_storage,
        )
        # 断言每个保存的对话消息长度为8
        assert len(saved_conversation.messages) == 8
        # 断言第一条保存的消息是用户消息类型
        assert isinstance(saved_conversation.messages[0], HumanMessage)
        # 断言第二条保存的消息是AI消息类型
        assert isinstance(saved_conversation.messages[1], AIMessage)
        # 断言第一条保存的消息内容为 "hello, this is first round"
        assert saved_conversation.messages[0].content == "hello, this is first round"
        # 断言第二条保存的消息内容为 "hi"
        assert saved_conversation.messages[1].content == "hi"


# 定义一个测试函数，用于测试带分页的查询对话功能
def test_query_with_page(
    conversation_list: List[StorageConversation], conv_storage, message_storage
):
    # 定义查询条件为用户名为 "user1" 的查询规格
    query_spec = QuerySpec(conditions={"user_name": "user1"})
    # 执行带分页的查询，并获取分页查询结果
    page_result: PaginationResult = conv_storage.paginate_query(
        page=1, page_size=2, cls=StorageConversation, spec=query_spec
    )
    # 断言分页查询结果的总数与对话列表长度相同
    assert page_result.total_count == len(conversation_list)
    # 断言分页查询结果的总页数为2
    assert page_result.total_pages == 2
    # 断言页面结果的页面大小为2，确保分页结果符合预期
    assert page_result.page_size == 2
    
    # 断言页面结果中的条目数量为2，确保分页结果中包含了两个条目
    assert len(page_result.items) == 2
    
    # 断言页面结果中第一个条目的会话唯一标识为"conv0"，确保第一个条目的属性值正确
    assert page_result.items[0].conv_uid == "conv0"
```