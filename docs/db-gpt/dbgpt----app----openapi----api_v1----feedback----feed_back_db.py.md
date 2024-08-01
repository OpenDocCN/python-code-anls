# `.\DB-GPT-src\dbgpt\app\openapi\api_v1\feedback\feed_back_db.py`

```py
# 导入需要的模块和类：日期时间处理模块 datetime，SQLAlchemy 中的列(Column)、日期时间类型(DateTime)、整数类型(Integer)、字符串类型(String)、文本类型(Text)
from datetime import datetime
from sqlalchemy import Column, DateTime, Integer, String, Text

# 导入自定义模块：反馈体类 FeedBackBody 和数据库访问基类 BaseDao，以及数据模型基类 Model
from dbgpt.app.openapi.api_v1.feedback.feed_back_model import FeedBackBody
from dbgpt.storage.metadata import BaseDao, Model

# ChatFeedBackEntity 类，继承自 Model 类，映射到数据库表 "chat_feed_back"
class ChatFeedBackEntity(Model):
    # 数据表名
    __tablename__ = "chat_feed_back"
    # 主键列 id，整数类型
    id = Column(Integer, primary_key=True)
    # 会话唯一标识 conv_uid，字符串类型，最大长度 128
    conv_uid = Column(String(128))
    # 会话索引 conv_index，整数类型
    conv_index = Column(Integer)
    # 分数 score，整数类型
    score = Column(Integer)
    # 问题类型 ques_type，字符串类型，最大长度 32
    ques_type = Column(String(32))
    # 问题内容 question，文本类型
    question = Column(Text)
    # 知识空间 knowledge_space，字符串类型，最大长度 128
    knowledge_space = Column(String(128))
    # 消息 messages，文本类型
    messages = Column(Text)
    # 用户名 user_name，字符串类型，最大长度 128
    user_name = Column(String(128))
    # 记录创建时间 gmt_created，日期时间类型
    gmt_created = Column(DateTime)
    # 记录修改时间 gmt_modified，日期时间类型
    gmt_modified = Column(DateTime)

    # 对象表示方法，用于打印对象时显示内容
    def __repr__(self):
        return (
            f"ChatFeekBackEntity(id={self.id}, conv_index='{self.conv_index}', conv_index='{self.conv_index}', "
            f"score='{self.score}', ques_type='{self.ques_type}', question='{self.question}', knowledge_space='{self.knowledge_space}', "
            f"messages='{self.messages}', user_name='{self.user_name}', gmt_created='{self.gmt_created}', gmt_modified='{self.gmt_modified}')"
        )

# ChatFeedBackDao 类，继承自 BaseDao 类，用于操作 ChatFeedBackEntity 模型对应的数据库表
class ChatFeedBackDao(BaseDao):
    # 创建或更新聊天反馈信息方法，接收一个 FeedBackBody 类型的参数 feed_back
    def create_or_update_chat_feed_back(self, feed_back: FeedBackBody):
        # 获取原始数据库会话对象
        session = self.get_raw_session()
        # 创建 ChatFeedBackEntity 对象，填充数据
        chat_feed_back = ChatFeedBackEntity(
            conv_uid=feed_back.conv_uid,
            conv_index=feed_back.conv_index,
            score=feed_back.score,
            ques_type=feed_back.ques_type,
            question=feed_back.question,
            knowledge_space=feed_back.knowledge_space,
            messages=feed_back.messages,
            user_name=feed_back.user_name,
            gmt_created=datetime.now(),  # 记录创建时间为当前时间
            gmt_modified=datetime.now(),  # 记录修改时间为当前时间
        )
        # 查询数据库是否存在指定 conv_uid 和 conv_index 的记录
        result = (
            session.query(ChatFeedBackEntity)
            .filter(ChatFeedBackEntity.conv_uid == feed_back.conv_uid)
            .filter(ChatFeedBackEntity.conv_index == feed_back.conv_index)
            .first()
        )
        # 如果存在记录，则更新记录的字段值
        if result is not None:
            result.score = feed_back.score
            result.ques_type = feed_back.ques_type
            result.question = feed_back.question
            result.knowledge_space = feed_back.knowledge_space
            result.messages = feed_back.messages
            result.user_name = feed_back.user_name
            result.gmt_created = datetime.now()  # 更新记录的创建时间为当前时间
            result.gmt_modified = datetime.now()  # 更新记录的修改时间为当前时间
        # 如果不存在记录，则将新创建的记录合并到会话中
        else:
            session.merge(chat_feed_back)
        # 提交会话中的所有操作
        session.commit()
        # 关闭数据库会话
        session.close()

    # 根据 conv_uid 和 conv_index 获取聊天反馈信息方法
    def get_chat_feed_back(self, conv_uid: str, conv_index: int):
        # 获取原始数据库会话对象
        session = self.get_raw_session()
        # 查询数据库中指定 conv_uid 和 conv_index 的记录
        result = (
            session.query(ChatFeedBackEntity)
            .filter(ChatFeedBackEntity.conv_uid == conv_uid)
            .filter(ChatFeedBackEntity.conv_index == conv_index)
            .first()
        )
        # 关闭数据库会话
        session.close()
        # 返回查询结果
        return result
```