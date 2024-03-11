# `.\Langchain-Chatchat\server\db\repository\message_repository.py`

```
# 从 server.db.session 模块导入 with_session 装饰器
# 从 typing 模块导入 Dict 和 List 类型
# 导入 uuid 模块
# 从 server.db.models.message_model 模块导入 MessageModel 类

# 使用 with_session 装饰器，将下面的函数包装在数据库会话中执行
def add_message_to_db(session, conversation_id: str, chat_type, query, response="", message_id=None,
                      metadata: Dict = {}):
    """
    新增聊天记录
    """
    # 如果没有提供 message_id，则生成一个新的 uuid
    if not message_id:
        message_id = uuid.uuid4().hex
    # 创建一个 MessageModel 对象，表示一条聊天记录
    m = MessageModel(id=message_id, chat_type=chat_type, query=query, response=response,
                     conversation_id=conversation_id,
                     meta_data=metadata)
    # 将该记录添加到数据库会话中
    session.add(m)
    # 提交会话，将更改保存到数据库
    session.commit()
    # 返回新增记录的 id
    return m.id

# 使用 with_session 装饰器，将下面的函数包装在数据库会话中执行
def update_message(session, message_id, response: str = None, metadata: Dict = None):
    """
    更新已有的聊天记录
    """
    # 根据 message_id 获取对应的聊天记录
    m = get_message_by_id(message_id)
    # 如果找到了对应记录
    if m is not None:
        # 如果提供了 response，则更新记录的 response 字段
        if response is not None:
            m.response = response
        # 如果提供了 metadata，并且是字典类型，则更新记录的 meta_data 字段
        if isinstance(metadata, dict):
            m.meta_data = metadata
        # 将更新后的记录添加到数据库会话中
        session.add(m)
        # 提交会话，将更改保存到数据库
        session.commit()
        # 返回更新记录的 id
        return m.id

# 使用 with_session 装饰器，将下面的函数包装在数据库会话中执行
def get_message_by_id(session, message_id) -> MessageModel:
    """
    查询聊天记录
    """
    # 根据 message_id 查询对应的聊天记录
    m = session.query(MessageModel).filter_by(id=message_id).first()
    # 返回查询到的聊天记录
    return m

# 使用 with_session 装饰器，将下面的函数包装在数据库会话中执行
def feedback_message_to_db(session, message_id, feedback_score, feedback_reason):
    """
    反馈聊天记录
    """
    # 根据 message_id 查询对应的聊天记录
    m = session.query(MessageModel).filter_by(id=message_id).first()
    # 如果找到了对应记录
    if m:
        # 更新记录的 feedback_score 和 feedback_reason 字段
        m.feedback_score = feedback_score
        m.feedback_reason = feedback_reason
    # 提交会话，将更改保存到数据库
    session.commit()
    # 返回更新记录的 id
    return m.id

# 使用 with_session 装饰器，将下面的函数包装在数据库会话中执行
def filter_message(session, conversation_id: str, limit: int = 10):
    # 查询指定 conversation_id 下的消息记录，按照创建时间倒序排序，限制返回数量为 limit
    messages = (session.query(MessageModel).filter_by(conversation_id=conversation_id).
                filter(MessageModel.response != '').
                order_by(MessageModel.create_time.desc()).limit(limit).all())
    # 初始化一个空列表用于存储查询结果
    data = []
    # 遍历消息列表中的每个消息对象
    for m in messages:
        # 将每个消息对象的查询和响应内容组成字典，添加到数据列表中
        data.append({"query": m.query, "response": m.response})
    # 返回包含所有消息查询和响应内容的列表
    return data
```