# `.\DB-GPT-src\dbgpt\app\knowledge\chunk_db.py`

```py
# 从 datetime 模块中导入 datetime 类，从 typing 模块中导入 List 类型
from datetime import datetime
from typing import List

# 从 sqlalchemy 模块中导入 Column, DateTime, Integer, String, Text 类，以及 func 函数
from sqlalchemy import Column, DateTime, Integer, String, Text, func

# 从 dbgpt._private.config 模块中导入 Config 类
from dbgpt._private.config import Config

# 从 dbgpt.serve.rag.api.schemas 模块中导入 DocumentChunkVO 类
from dbgpt.serve.rag.api.schemas import DocumentChunkVO

# 从 dbgpt.storage.metadata 模块中导入 BaseDao 类和 Model 类
from dbgpt.storage.metadata import BaseDao, Model

# 创建一个 Config 对象，赋值给 CFG 变量
CFG = Config()

# 定义 DocumentChunkEntity 类，继承自 Model 类
class DocumentChunkEntity(Model):
    # 设置数据表名为 "document_chunk"
    __tablename__ = "document_chunk"
    
    # 定义 id 列，主键，整数类型
    id = Column(Integer, primary_key=True)
    # 定义 document_id 列，整数类型
    document_id = Column(Integer)
    # 定义 doc_name 列，字符串类型，最大长度为 100
    doc_name = Column(String(100))
    # 定义 doc_type 列，字符串类型，最大长度为 100
    doc_type = Column(String(100))
    # 定义 content 列，文本类型
    content = Column(Text)
    # 定义 meta_info 列，字符串类型，最大长度为 500
    meta_info = Column(String(500))
    # 定义 gmt_created 列，日期时间类型
    gmt_created = Column(DateTime)
    # 定义 gmt_modified 列，日期时间类型
    gmt_modified = Column(DateTime)
    
    # 定义 __repr__ 方法，返回实例的字符串表示，包含各列的值
    def __repr__(self):
        return f"DocumentChunkEntity(id={self.id}, doc_name='{self.doc_name}', doc_type='{self.doc_type}', document_id='{self.document_id}', content='{self.content}', meta_info='{self.meta_info}', gmt_created='{self.gmt_created}', gmt_modified='{self.gmt_modified}')"

    # 定义类方法 to_to_document_chunk_vo，将 DocumentChunkEntity 实例列表转换为 DocumentChunkVO 实例列表
    @classmethod
    def to_to_document_chunk_vo(cls, entity_list: List["DocumentChunkEntity"]):
        return [
            DocumentChunkVO(
                id=entity.id,
                document_id=entity.document_id,
                doc_name=entity.doc_name,
                doc_type=entity.doc_type,
                content=entity.content,
                meta_info=entity.meta_info,
                gmt_created=entity.gmt_created.strftime("%Y-%m-%d %H:%M:%S"),
                gmt_modified=entity.gmt_modified.strftime("%Y-%m-%d %H:%M:%S"),
            )
            for entity in entity_list
        ]


# 定义 DocumentChunkDao 类，继承自 BaseDao 类
class DocumentChunkDao(BaseDao):
    # 定义 create_documents_chunks 方法，创建多个文档块实例并添加到数据库中
    def create_documents_chunks(self, documents: List):
        # 获取数据库会话对象
        session = self.get_raw_session()
        # 使用列表推导式创建 DocumentChunkEntity 实例列表
        docs = [
            DocumentChunkEntity(
                doc_name=document.doc_name,
                doc_type=document.doc_type,
                document_id=document.document_id,
                content=document.content or "",
                meta_info=document.meta_info or "",
                gmt_created=datetime.now(),
                gmt_modified=datetime.now(),
            )
            for document in documents
        ]
        # 将实例列表添加到会话中
        session.add_all(docs)
        # 提交会话的事务
        session.commit()
        # 关闭会话
        session.close()

    # 定义 get_document_chunks 方法，查询文档块实例
    def get_document_chunks(
        self, query: DocumentChunkEntity, page=1, page_size=20, document_ids=None
    ):
        # 方法体暂未提供，需进一步完善
        pass
    # 返回 DocumentChunkVO 对象列表
    def get_document_chunks(
        self,
        query: DocumentChunkEntity,
        page: int,
        page_size: int,
        document_ids: Optional[List[int]]
    ) -> List[DocumentChunkVO]:
        # 获取原始数据库会话
        session = self.get_raw_session()
        
        # 从数据库中查询 DocumentChunkEntity 实体
        document_chunks = session.query(DocumentChunkEntity)
        
        # 根据查询条件过滤结果集
        if query.id is not None:
            document_chunks = document_chunks.filter(DocumentChunkEntity.id == query.id)
        if query.document_id is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.document_id == query.document_id
            )
        if query.doc_type is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.doc_type == query.doc_type
            )
        if query.content is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.content == query.content
            )
        if query.doc_name is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.doc_name == query.doc_name
            )
        if query.meta_info is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.meta_info == query.meta_info
            )
        if document_ids is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.document_id.in_(document_ids)
            )
        
        # 按照 DocumentChunkEntity 的 id 升序排列
        document_chunks = document_chunks.order_by(DocumentChunkEntity.id.asc())
        
        # 根据分页参数获取指定页的数据，并限制每页数据量
        document_chunks = document_chunks.offset((page - 1) * page_size).limit(
            page_size
        )
        
        # 执行查询并获取所有结果
        result = document_chunks.all()
        
        # 关闭数据库会话
        session.close()
        
        # 将查询结果转换为 DocumentChunkVO 对象列表并返回
        return DocumentChunkEntity.to_to_document_chunk_vo(result)

    # 返回符合查询条件的 DocumentChunkEntity 记录数量
    def get_document_chunks_count(self, query: DocumentChunkEntity):
        # 获取原始数据库会话
        session = self.get_raw_session()
        
        # 查询符合条件的 DocumentChunkEntity 记录数量
        document_chunks = session.query(func.count(DocumentChunkEntity.id))
        
        # 根据查询条件过滤结果集
        if query.id is not None:
            document_chunks = document_chunks.filter(DocumentChunkEntity.id == query.id)
        if query.document_id is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.document_id == query.document_id
            )
        if query.doc_type is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.doc_type == query.doc_type
            )
        if query.doc_name is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.doc_name == query.doc_name
            )
        if query.meta_info is not None:
            document_chunks = document_chunks.filter(
                DocumentChunkEntity.meta_info == query.meta_info
            )
        
        # 执行聚合查询，获取记录数量
        count = document_chunks.scalar()
        
        # 关闭数据库会话
        session.close()
        
        # 返回符合条件的 DocumentChunkEntity 记录数量
        return count
    # 定义一个方法，用于从数据库中删除指定 document_id 对应的记录
    def raw_delete(self, document_id: int):
        # 获取一个数据库会话对象
        session = self.get_raw_session()
        
        # 如果 document_id 为 None，则抛出异常
        if document_id is None:
            raise Exception("document_id is None")
        
        # 创建一个查询对象，用于查询指定 document_id 的文档块实体
        query = DocumentChunkEntity(document_id=document_id)
        
        # 从数据库中查询所有符合条件的文档块实体
        knowledge_documents = session.query(DocumentChunkEntity)
        
        # 如果查询对象中的 document_id 不为 None，则过滤出与之匹配的文档块
        if query.document_id is not None:
            chunks = knowledge_documents.filter(
                DocumentChunkEntity.document_id == query.document_id
            )
        
        # 执行删除操作，删除所有符合条件的文档块
        chunks.delete()
        
        # 提交事务，确保删除操作生效
        session.commit()
        
        # 关闭数据库会话
        session.close()
```