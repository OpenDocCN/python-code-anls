# `.\Langchain-Chatchat\server\db\models\message_model.py`

```
# 导入需要的模块
from sqlalchemy import Column, Integer, String, DateTime, JSON, func

# 导入基类Base
from server.db.base import Base

# 定义消息模型类
class MessageModel(Base):
    """
    聊天记录模型
    """
    # 指定表名
    __tablename__ = 'message'
    # 聊天记录ID，主键
    id = Column(String(32), primary_key=True, comment='聊天记录ID')
    # 对话框ID
    conversation_id = Column(String(32), default=None, index=True, comment='对话框ID')
    # 聊天类型
    chat_type = Column(String(50), comment='聊天类型')
    # 用户问题
    query = Column(String(4096), comment='用户问题')
    # 模型回答
    response = Column(String(4096), comment='模型回答')
    # 记录知识库id等，以便后续扩展
    meta_data = Column(JSON, default={})
    # 用户评分，满分100 越高表示评价越好
    feedback_score = Column(Integer, default=-1, comment='用户评分')
    # 用户评分理由
    feedback_reason = Column(String(255), default="", comment='用户评分理由')
    # 创建时间
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    # 定义__repr__方法，用于返回对象的字符串表示
    def __repr__(self):
        return f"<message(id='{self.id}', conversation_id='{self.conversation_id}', chat_type='{self.chat_type}', query='{self.query}', response='{self.response}',meta_data='{self.meta_data}',feedback_score='{self.feedback_score}',feedback_reason='{self.feedback_reason}', create_time='{self.create_time}')>"
```