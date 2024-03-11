# `.\Langchain-Chatchat\server\db\models\knowledge_base_model.py`

```py
# 导入需要的模块
from sqlalchemy import Column, Integer, String, DateTime, func

# 导入基类
from server.db.base import Base

# 定义知识库模型类
class KnowledgeBaseModel(Base):
    """
    知识库模型
    """
    # 指定表名
    __tablename__ = 'knowledge_base'
    
    # 定义主键字段
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识库ID')
    
    # 定义知识库名称字段
    kb_name = Column(String(50), comment='知识库名称')
    
    # 定义知识库简介字段
    kb_info = Column(String(200), comment='知识库简介(用于Agent)')
    
    # 定义向量库类型字段
    vs_type = Column(String(50), comment='向量库类型')
    
    # 定义嵌入模型名称字段
    embed_model = Column(String(50), comment='嵌入模型名称')
    
    # 定义文件数量字段，默认值为0
    file_count = Column(Integer, default=0, comment='文件数量')
    
    # 定义创建时间字段，默认为当前时间
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    # 定义对象的字符串表示形式
    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', kb_name='{self.kb_name}',kb_intro='{self.kb_info} vs_type='{self.vs_type}', embed_model='{self.embed_model}', file_count='{self.file_count}', create_time='{self.create_time}')>"
```