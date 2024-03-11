# `.\Langchain-Chatchat\server\db\repository\conversation_repository.py`

```
# 从 server.db.session 模块中导入 with_session 装饰器
from server.db.session import with_session
# 导入 uuid 模块
import uuid
# 从 server.db.models.conversation_model 模块中导入 ConversationModel 类

# 使用 with_session 装饰器，确保数据库会话的正确处理
@with_session
# 定义一个函数，用于向数据库中添加聊天记录
def add_conversation_to_db(session, chat_type, name="", conversation_id=None):
    """
    新增聊天记录
    """
    # 如果没有提供 conversation_id，则生成一个新的唯一标识符
    if not conversation_id:
        conversation_id = uuid.uuid4().hex
    # 创建一个 ConversationModel 对象，传入聊天类型和名称
    c = ConversationModel(id=conversation_id, chat_type=chat_type, name=name)

    # 将 ConversationModel 对象添加到数据库会话中
    session.add(c)
    # 返回添加的聊天记录的唯一标识符
    return c.id
```