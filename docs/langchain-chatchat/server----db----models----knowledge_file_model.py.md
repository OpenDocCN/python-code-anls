# `.\Langchain-Chatchat\server\db\models\knowledge_file_model.py`

```py
# 导入需要的模块和类
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, JSON, func
# 导入自定义的 Base 类
from server.db.base import Base

# 定义知识文件模型类
class KnowledgeFileModel(Base):
    """
    知识文件模型
    """
    # 指定表名
    __tablename__ = 'knowledge_file'
    # 定义主键字段
    id = Column(Integer, primary_key=True, autoincrement=True, comment='知识文件ID')
    # 定义文件名字段
    file_name = Column(String(255), comment='文件名')
    # 定义文件扩展名字段
    file_ext = Column(String(10), comment='文件扩展名')
    # 定义所属知识库名称字段
    kb_name = Column(String(50), comment='所属知识库名称')
    # 定义文档加载器名称字段
    document_loader_name = Column(String(50), comment='文档加载器名称')
    # 定义文本分割器名称字段
    text_splitter_name = Column(String(50), comment='文本分割器名称')
    # 定义文件版本字段，默认值为1
    file_version = Column(Integer, default=1, comment='文件版本')
    # 定义文件修改时间字段，默认值为0.0
    file_mtime = Column(Float, default=0.0, comment="文件修改时间")
    # 定义文件大小字段，默认值为0
    file_size = Column(Integer, default=0, comment="文件大小")
    # 定义是否自定义docs字段，默认值为False
    custom_docs = Column(Boolean, default=False, comment="是否自定义docs")
    # 定义切分文档数量字段，默认值为0
    docs_count = Column(Integer, default=0, comment="切分文档数量")
    # 定义创建时间字段，使用数据库函数 func.now() 获取当前时间
    create_time = Column(DateTime, default=func.now(), comment='创建时间')

    # 定义 __repr__ 方法，返回对象的字符串表示形式
    def __repr__(self):
        return f"<KnowledgeFile(id='{self.id}', file_name='{self.file_name}', file_ext='{self.file_ext}', kb_name='{self.kb_name}', document_loader_name='{self.document_loader_name}', text_splitter_name='{self.text_splitter_name}', file_version='{self.file_version}', create_time='{self.create_time}')>"

# 定义文件-向量库文档模型类
class FileDocModel(Base):
    """
    文件-向量库文档模型
    """
    # 指定表名
    __tablename__ = 'file_doc'
    # 定义主键字段
    id = Column(Integer, primary_key=True, autoincrement=True, comment='ID')
    # 定义知识库名称字段
    kb_name = Column(String(50), comment='知识库名称')
    # 定义文件名称字段
    file_name = Column(String(255), comment='文件名称')
    # 定义向量库文档ID字段
    doc_id = Column(String(50), comment="向量库文档ID")
    # 定义元数据字段，类型为 JSON，默认为空字典
    meta_data = Column(JSON, default={})

    # 定义 __repr__ 方法，返回对象的字符串表示形式
    def __repr__(self):
        return f"<FileDoc(id='{self.id}', kb_name='{self.kb_name}', file_name='{self.file_name}', doc_id='{self.doc_id}', metadata='{self.meta_data}')>"
```