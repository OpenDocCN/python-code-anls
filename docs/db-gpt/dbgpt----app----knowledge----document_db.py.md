# `.\DB-GPT-src\dbgpt\app\knowledge\document_db.py`

```py
from datetime import datetime  # 导入 datetime 模块中的 datetime 类
from typing import Any, Dict, List, Union  # 导入类型提示相关的模块

from sqlalchemy import Column, DateTime, Integer, String, Text, func  # 导入 SQLAlchemy 相关模块

from dbgpt._private.config import Config  # 导入配置文件类 Config
from dbgpt._private.pydantic import model_to_dict  # 导入将模型转换为字典的函数
from dbgpt.serve.conversation.api.schemas import ServeRequest  # 导入对话 API 的请求模式
from dbgpt.serve.rag.api.schemas import (  # 导入 RAG API 的请求和响应模式
    DocumentServeRequest,
    DocumentServeResponse,
    DocumentVO,
)
from dbgpt.storage.metadata import BaseDao, Model  # 导入存储元数据的基础 DAO 和模型类

CFG = Config()  # 创建 Config 类的实例，用于获取配置信息


class KnowledgeDocumentEntity(Model):
    __tablename__ = "knowledge_document"  # 定义数据库表名为 'knowledge_document'
    id = Column(Integer, primary_key=True)  # 定义整型主键 id
    doc_name = Column(String(100))  # 定义字符串字段 doc_name，长度限制为 100
    doc_type = Column(String(100))  # 定义字符串字段 doc_type，长度限制为 100
    space = Column(String(100))  # 定义字符串字段 space，长度限制为 100
    chunk_size = Column(Integer)  # 定义整型字段 chunk_size
    status = Column(String(100))  # 定义字符串字段 status，长度限制为 100
    last_sync = Column(DateTime)  # 定义日期时间字段 last_sync
    content = Column(Text)  # 定义文本字段 content
    result = Column(Text)  # 定义文本字段 result
    vector_ids = Column(Text)  # 定义文本字段 vector_ids
    summary = Column(Text)  # 定义文本字段 summary
    gmt_created = Column(DateTime)  # 定义日期时间字段 gmt_created
    gmt_modified = Column(DateTime)  # 定义日期时间字段 gmt_modified

    def __repr__(self):
        # 返回实例的可打印字符串表示，包含主要字段的信息
        return f"KnowledgeDocumentEntity(id={self.id}, doc_name='{self.doc_name}', doc_type='{self.doc_type}', chunk_size='{self.chunk_size}', status='{self.status}', last_sync='{self.last_sync}', content='{self.content}', result='{self.result}', summary='{self.summary}', gmt_created='{self.gmt_created}', gmt_modified='{self.gmt_modified}')"

    @classmethod
    def to_document_vo(
        cls, entity_list: List["KnowledgeDocumentEntity"]
    ) -> List[DocumentVO]:
        vo_results = []  # 初始化空列表，用于存储转换后的 DocumentVO 对象
        for item in entity_list:
            vo_results.append(
                DocumentVO(
                    id=item.id,
                    doc_name=item.doc_name,
                    doc_type=item.doc_type,
                    space=item.space,
                    chunk_size=item.chunk_size,
                    status=item.status,
                    last_sync=item.last_sync.strftime("%Y-%m-%d %H:%M:%S"),  # 格式化日期时间字段为字符串
                    content=item.content,
                    result=item.result,
                    vector_ids=item.vector_ids,
                    summary=item.summary,
                    gmt_created=item.gmt_created.strftime("%Y-%m-%d %H:%M:%S"),  # 格式化日期时间字段为字符串
                    gmt_modified=item.gmt_modified.strftime("%Y-%m-%d %H:%M:%S"),  # 格式化日期时间字段为字符串
                )
            )
        return vo_results  # 返回转换后的 DocumentVO 对象列表

    @classmethod
    # 根据给定的 DocumentVO 对象创建一个 KnowledgeDocumentEntity 实例
    def from_document_vo(cls, vo: DocumentVO) -> "KnowledgeDocumentEntity":
        # 使用 DocumentVO 对象的属性初始化 KnowledgeDocumentEntity 实例
        entity = KnowledgeDocumentEntity(
            id=vo.id,                # 设置实体的 id
            doc_name=vo.doc_name,    # 设置实体的 doc_name
            doc_type=vo.doc_type,    # 设置实体的 doc_type
            space=vo.space,          # 设置实体的 space
            chunk_size=vo.chunk_size,  # 设置实体的 chunk_size
            status=vo.status,        # 设置实体的 status
            content=vo.content,      # 设置实体的 content
            result=vo.result,        # 设置实体的 result
            vector_ids=vo.vector_ids,  # 设置实体的 vector_ids
            summary=vo.summary,      # 设置实体的 summary
        )
        # 如果 DocumentVO 对象有 last_sync 属性，则将其转换为 datetime 对象并设置为实体的 last_sync 属性
        if vo.last_sync:
            entity.last_sync = datetime.strptime(vo.last_sync, "%Y-%m-%d %H:%M:%S")
        # 如果 DocumentVO 对象有 gmt_created 属性，则将其转换为 datetime 对象并设置为实体的 gmt_created 属性
        if vo.gmt_created:
            entity.gmt_created = datetime.strptime(vo.gmt_created, "%Y-%m-%d %H:%M:%S")
        # 如果 DocumentVO 对象有 gmt_modified 属性，则将其转换为 datetime 对象并设置为实体的 gmt_modified 属性
        if vo.gmt_modified:
            entity.gmt_modified = datetime.strptime(vo.gmt_modified, "%Y-%m-%d %H:%M:%S")
        # 返回创建的 KnowledgeDocumentEntity 实例
        return entity
# KnowledgeDocumentDao 类，继承自 BaseDao 类
class KnowledgeDocumentDao(BaseDao):

    # 创建知识文档方法
    def create_knowledge_document(self, document: KnowledgeDocumentEntity):
        # 获取原始数据库会话
        session = self.get_raw_session()

        # 创建 KnowledgeDocumentEntity 实例
        knowledge_document = KnowledgeDocumentEntity(
            doc_name=document.doc_name,
            doc_type=document.doc_type,
            space=document.space,
            chunk_size=0.0,
            status=document.status,
            last_sync=document.last_sync,
            content=document.content or "",
            result=document.result or "",
            vector_ids=document.vector_ids,
            gmt_created=datetime.now(),
            gmt_modified=datetime.now(),
        )

        # 将实例添加到会话中
        session.add(knowledge_document)

        # 提交会话的事务
        session.commit()

        # 获取新创建文档的 ID
        doc_id = knowledge_document.id

        # 关闭会话
        session.close()

        # 返回新创建文档的 ID
        return doc_id

    # 获取知识文档列表方法
    def get_knowledge_documents(self, query, page=1, page_size=20) -> List[DocumentVO]:
        """获取与给定查询匹配的文档列表。
        Args:
            query: 包含查询参数的 KnowledgeDocumentEntity 对象。
            page: 要返回的页码。
            page_size: 每页返回的文档数量。
        """
        # 获取原始数据库会话
        session = self.get_raw_session()

        # 打印当前会话信息
        print(f"current session:{session}")

        # 查询 KnowledgeDocumentEntity 对象
        knowledge_documents = session.query(KnowledgeDocumentEntity)

        # 根据查询条件过滤文档列表
        if query.id is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.id == query.id
            )
        if query.doc_name is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.doc_name == query.doc_name
            )
        if query.doc_type is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.doc_type == query.doc_type
            )
        if query.space is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.space == query.space
            )
        if query.status is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.status == query.status
            )

        # 按文档 ID 降序排序
        knowledge_documents = knowledge_documents.order_by(
            KnowledgeDocumentEntity.id.desc()
        )

        # 分页查询文档列表
        knowledge_documents = knowledge_documents.offset((page - 1) * page_size).limit(
            page_size
        )

        # 获取查询结果列表
        result = knowledge_documents.all()

        # 关闭会话
        session.close()

        # 将查询结果转换为 DocumentVO 类型并返回
        return KnowledgeDocumentEntity.to_document_vo(result)
    def documents_by_ids(self, ids) -> List[DocumentVO]:
        """Get a list of documents by their IDs.
        Args:
            ids: A list of document IDs.
        Returns:
            A list of KnowledgeDocumentEntity objects.
        """
        # 获取一个新的数据库会话对象
        session = self.get_raw_session()
        # 打印当前会话对象的信息，用于调试或日志记录
        print(f"current session:{session}")
        # 创建一个查询对象，选择所有符合条件的 KnowledgeDocumentEntity 实例
        knowledge_documents = session.query(KnowledgeDocumentEntity)
        # 根据传入的 IDs 过滤查询结果
        knowledge_documents = knowledge_documents.filter(
            KnowledgeDocumentEntity.id.in_(ids)
        )
        # 获取所有符合条件的结果列表
        result = knowledge_documents.all()
        # 关闭数据库会话
        session.close()
        # 将查询结果转换为 DocumentVO 对象的列表并返回
        return KnowledgeDocumentEntity.to_document_vo(result)

    def get_documents(self, query):
        # 获取一个新的数据库会话对象
        session = self.get_raw_session()
        # 打印当前会话对象的信息，用于调试或日志记录
        print(f"current session:{session}")
        # 创建一个查询对象，选择所有符合条件的 KnowledgeDocumentEntity 实例
        knowledge_documents = session.query(KnowledgeDocumentEntity)
        # 根据查询对象的属性进行过滤
        if query.id is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.id == query.id
            )
        if query.doc_name is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.doc_name == query.doc_name
            )
        if query.doc_type is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.doc_type == query.doc_type
            )
        if query.space is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.space == query.space
            )
        if query.status is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.status == query.status
            )
        # 根据 ID 降序排序结果
        knowledge_documents = knowledge_documents.order_by(
            KnowledgeDocumentEntity.id.desc()
        )
        # 获取所有符合条件的结果列表
        result = knowledge_documents.all()
        # 关闭数据库会话
        session.close()
        # 返回查询结果列表
        return result

    def get_knowledge_documents_count_bulk(self, space_names):
        # 获取一个新的数据库会话对象
        session = self.get_raw_session()
        """
        Perform a batch query to count the number of documents for each knowledge space.

        Args:
            space_names: A list of knowledge space names to query for document counts.
            session: A SQLAlchemy session object.

        Returns:
            A dictionary mapping each space name to its document count.
        """
        # 创建一个查询，统计每个知识空间中的文档数目
        counts_query = (
            session.query(
                KnowledgeDocumentEntity.space,
                func.count(KnowledgeDocumentEntity.id).label("document_count"),
            )
            .filter(KnowledgeDocumentEntity.space.in_(space_names))
            .group_by(KnowledgeDocumentEntity.space)
        )

        # 获取所有符合条件的结果列表
        results = counts_query.all()
        # 关闭数据库会话
        session.close()
        # 创建一个字典，将每个空间名映射到其对应的文档数目，并返回该字典
        docs_count = {result.space: result.document_count for result in results}
        return docs_count
    # 获取知识文档的数量，根据给定的查询条件
    def get_knowledge_documents_count(self, query):
        # 获取原始数据库会话
        session = self.get_raw_session()
        # 查询知识文档数量，使用聚合函数 func.count() 统计 KnowledgeDocumentEntity 表中 id 的数量
        knowledge_documents = session.query(func.count(KnowledgeDocumentEntity.id))
        # 如果查询对象有指定 id，则添加 id 的过滤条件
        if query.id is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.id == query.id
            )
        # 如果查询对象有指定 doc_name，则添加 doc_name 的过滤条件
        if query.doc_name is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.doc_name == query.doc_name
            )
        # 如果查询对象有指定 doc_type，则添加 doc_type 的过滤条件
        if query.doc_type is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.doc_type == query.doc_type
            )
        # 如果查询对象有指定 space，则添加 space 的过滤条件
        if query.space is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.space == query.space
            )
        # 如果查询对象有指定 status，则添加 status 的过滤条件
        if query.status is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.status == query.status
            )
        # 获取最终的文档数量结果
        count = knowledge_documents.scalar()
        # 关闭数据库会话
        session.close()
        # 返回文档数量
        return count

    # 更新知识文档
    def update_knowledge_document(self, document: KnowledgeDocumentEntity):
        # 获取原始数据库会话
        session = self.get_raw_session()
        # 合并更新文档到数据库会话中
        updated_space = session.merge(document)
        # 提交事务，更新数据库
        session.commit()
        # 获取更新后的文档的 id
        update_space_id = updated_space.id
        # 关闭数据库会话
        session.close()
        # 返回更新后文档的 id
        return update_space_id

    # 根据查询条件删除知识文档
    def raw_delete(self, query: KnowledgeDocumentEntity):
        # 获取原始数据库会话
        session = self.get_raw_session()
        # 查询需要删除的知识文档对象
        knowledge_documents = session.query(KnowledgeDocumentEntity)
        # 如果查询对象有指定 id，则添加 id 的过滤条件
        if query.id is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.id == query.id
            )
        # 如果查询对象有指定 doc_name，则添加 doc_name 的过滤条件
        if query.doc_name is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.doc_name == query.doc_name
            )
        # 如果查询对象有指定 space，则添加 space 的过滤条件
        if query.space is not None:
            knowledge_documents = knowledge_documents.filter(
                KnowledgeDocumentEntity.space == query.space
            )
        # 执行删除操作
        knowledge_documents.delete()
        # 提交事务，更新数据库
        session.commit()
        # 关闭数据库会话
        session.close()

    # 根据请求对象创建知识文档实体
    def from_request(
        self, request: Union[ServeRequest, Dict[str, Any]]
    ) -> KnowledgeDocumentEntity:
        """Convert the request to an entity

        Args:
            request (Union[ServeRequest, Dict[str, Any]]): The request

        Returns:
            T: The entity
        """
        # 如果请求是 DocumentServeRequest 类型，则转换成字典形式
        request_dict = (
            model_to_dict(request)
            if isinstance(request, DocumentServeRequest)
            else request
        )
        # 根据转换后的字典创建 KnowledgeDocumentEntity 实体对象
        entity = KnowledgeDocumentEntity(**request_dict)
        # 返回创建的实体对象
        return entity
    def to_request(self, entity: KnowledgeDocumentEntity) -> DocumentServeResponse:
        """Convert the entity to a request

        Args:
            entity (T): The entity to be converted

        Returns:
            REQ: The request object representing the converted entity
        """
        # 创建一个新的 DocumentServeResponse 对象，使用给定实体的属性填充
        return DocumentServeResponse(
            id=entity.id,
            doc_name=entity.doc_name,
            doc_type=entity.doc_type,
            space=entity.space,
            chunk_size=entity.chunk_size,
            status=entity.status,
            last_sync=entity.last_sync,
            content=entity.content,
            result=entity.result,
            vector_ids=entity.vector_ids,
            summary=entity.summary,
            gmt_created=entity.gmt_created,
            gmt_modified=entity.gmt_modified,
        )

    def to_response(self, entity: KnowledgeDocumentEntity) -> DocumentServeResponse:
        """Convert the entity to a response

        Args:
            entity (T): The entity to be converted

        Returns:
            REQ: The response object representing the converted entity
        """
        # 创建一个新的 DocumentServeResponse 对象，使用给定实体的属性填充
        return DocumentServeResponse(
            id=entity.id,
            doc_name=entity.doc_name,
            doc_type=entity.doc_type,
            space=entity.space,
            chunk_size=entity.chunk_size,
            status=entity.status,
            last_sync=entity.last_sync,
            content=entity.content,
            result=entity.result,
            vector_ids=entity.vector_ids,
            summary=entity.summary,
            gmt_created=entity.gmt_created,
            gmt_modified=entity.gmt_modified,
        )
```