# `.\Langchain-Chatchat\server\db\models\conversation_model.py`

```
# 导入需要的模块和类
from sqlalchemy import Column, Integer, String, DateTime, JSON, func
from server.db.base import Base

# 定义聊天记录模型类
class ConversationModel(Base):
    """
    聊天记录模型
    """
    # 指定数据库表名
    __tablename__ = 'conversation'
    # 对话框ID，主键
    id = Column(String(32), primary_key=True, comment='对话框ID')
    # 对话框名称
    name = Column(String(50), comment='对话框名称')
    # 聊天类型，如chat/agent_chat等
    chat_type = Column(String(50), comment='聊天类型')
    # 创建时间，默认为当前时间
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    # 定义对象的字符串表示形式
    def __repr__(self):
        return f"<Conversation(id='{self.id}', name='{self.name}', chat_type='{self.chat_type}', create_time='{self.create_time}')>"
```