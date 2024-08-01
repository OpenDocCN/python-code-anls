# `.\DB-GPT-src\dbgpt\serve\rag\service\service.py`

```py
# 引入异步IO模块
import asyncio
# 引入JSON处理模块
import json
# 引入日志记录模块
import logging
# 引入操作系统相关功能模块
import os
# 引入文件和目录操作模块
import shutil
# 引入临时文件和目录创建模块
import tempfile
# 引入日期时间处理模块
from datetime import datetime
# 引入枚举类型定义模块
from enum import Enum
# 引入类型提示相关模块
from typing import List, Optional, cast

# 引入FastAPI的HTTP异常类
from fastapi import HTTPException

# 引入私有配置模块
from dbgpt._private.config import Config
# 引入文档块数据库相关模块
from dbgpt.app.knowledge.chunk_db import DocumentChunkDao, DocumentChunkEntity
# 引入知识文档数据库相关模块
from dbgpt.app.knowledge.document_db import KnowledgeDocumentDao, KnowledgeDocumentEntity
# 引入知识空间请求相关模块
from dbgpt.app.knowledge.request.request import BusinessFieldType, KnowledgeSpaceRequest
# 引入组件类型和系统应用相关模块
from dbgpt.component import ComponentType, SystemApp
# 引入知识工厂域类型键相关配置
from dbgpt.configs import TAG_KEY_KNOWLEDGE_FACTORY_DOMAIN_TYPE
# 引入嵌入模型配置和知识上传根路径配置
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG, KNOWLEDGE_UPLOAD_ROOT_PATH
# 引入LLM客户端相关模块
from dbgpt.core import LLMClient
# 引入默认LLM客户端模块
from dbgpt.model import DefaultLLMClient
# 引入工作管理器工厂模块
from dbgpt.model.cluster import WorkerManagerFactory
# 引入RAG组装器相关模块
from dbgpt.rag.assembler import EmbeddingAssembler
# 引入文档块管理器相关模块
from dbgpt.rag.chunk_manager import ChunkParameters
# 引入嵌入工厂相关模块
from dbgpt.rag.embedding import EmbeddingFactory
# 引入知识工厂相关模块
from dbgpt.rag.knowledge import ChunkStrategy, KnowledgeFactory, KnowledgeType
# 引入基础服务类模块
from dbgpt.serve.core import BaseService
# 引入向量存储连接器模块
from dbgpt.serve.rag.connector import VectorStoreConnector
# 引入元数据存储基础DAO模块
from dbgpt.storage.metadata import BaseDao
# 引入元数据存储基础DAO查询规范模块
from dbgpt.storage.metadata._base_dao import QUERY_SPEC
# 引入向量存储配置模块
from dbgpt.storage.vector_store.base import VectorStoreConfig
# 引入执行器工厂模块
from dbgpt.util.executor_utils import ExecutorFactory, blocking_func_to_async
# 引入分页工具模块
from dbgpt.util.pagination_utils import PaginationResult
# 引入跟踪器相关模块
from dbgpt.util.tracer import root_tracer, trace

# 引入本地的API模型schemas
from ..api.schemas import (
    DocumentServeRequest,
    DocumentServeResponse,
    DocumentVO,
    KnowledgeSyncRequest,
    SpaceServeRequest,
    SpaceServeResponse,
)
# 引入服务的配置相关模块
from ..config import SERVE_CONFIG_KEY_PREFIX, SERVE_SERVICE_COMPONENT_NAME, ServeConfig
# 引入知识空间DAO和实体模块
from ..models.models import KnowledgeSpaceDao, KnowledgeSpaceEntity

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 获取私有配置对象实例
CFG = Config()

# 定义同步状态枚举
class SyncStatus(Enum):
    TODO = "TODO"
    FAILED = "FAILED"
    RUNNING = "RUNNING"
    FINISHED = "FINISHED"

# 定义服务类，继承自BaseService，用于处理知识空间相关请求
class Service(BaseService[KnowledgeSpaceEntity, SpaceServeRequest, SpaceServeResponse]):
    """The service class for Flow"""

    # 类属性：服务组件的名称
    name = SERVE_SERVICE_COMPONENT_NAME

    # 初始化方法，接收系统应用、DAO对象作为参数
    def __init__(
        self,
        system_app: SystemApp,
        dao: Optional[KnowledgeSpaceDao] = None,
        document_dao: Optional[KnowledgeDocumentDao] = None,
        chunk_dao: Optional[DocumentChunkDao] = None,
    ):
        # 调用父类的初始化方法
        super().__init__(system_app)
        # 存储系统应用实例
        self._system_app = system_app
        # 存储知识空间DAO实例
        self._dao: KnowledgeSpaceDao = dao
        # 存储知识文档DAO实例
        self._document_dao: KnowledgeDocumentDao = document_dao
        # 存储文档块DAO实例
        self._chunk_dao: DocumentChunkDao = chunk_dao
    def init_app(self, system_app: SystemApp) -> None:
        """Initialize the service
        
        Args:
            system_app (SystemApp): The system app instance to initialize with
        """
        # 调用父类方法初始化应用
        super().init_app(system_app)
        # 从系统应用配置中获取服务配置信息
        self._serve_config = ServeConfig.from_app_config(
            system_app.config, SERVE_CONFIG_KEY_PREFIX
        )
        # 初始化 DAO 对象，如果未提供则使用默认的 KnowledgeSpaceDao
        self._dao = self._dao or KnowledgeSpaceDao()
        # 初始化文档 DAO 对象，如果未提供则使用默认的 KnowledgeDocumentDao
        self._document_dao = self._document_dao or KnowledgeDocumentDao()
        # 初始化文档块 DAO 对象，如果未提供则使用默认的 DocumentChunkDao
        self._chunk_dao = self._chunk_dao or DocumentChunkDao()
        # 存储系统应用实例
        self._system_app = system_app

    @property
    def dao(
        self,
    ) -> BaseDao[KnowledgeSpaceEntity, SpaceServeRequest, SpaceServeResponse]:
        """Returns the internal DAO instance."""
        # 返回内部 DAO 对象
        return self._dao

    @property
    def config(self) -> ServeConfig:
        """Returns the internal ServeConfig instance."""
        # 返回内部服务配置对象
        return self._serve_config

    @property
    def llm_client(self) -> LLMClient:
        # 获取 WorkerManagerFactory 实例并创建工作管理器
        worker_manager = self._system_app.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        # 返回使用默认工作管理器的 LLM 客户端
        return DefaultLLMClient(worker_manager, True)

    def create_space(self, request: SpaceServeRequest) -> SpaceServeResponse:
        """Create a new Space entity
        
        Args:
            request (SpaceServeRequest): The request object containing space information
        
        Returns:
            SpaceServeResponse: The response object indicating success or failure
        """
        # 检查是否已存在同名空间，如果存在则抛出异常
        space = self.get(request)
        if space is not None:
            raise HTTPException(
                status_code=400,
                detail=f"space name:{request.name} have already named",
            )
        # 创建知识空间并返回创建结果
        return self._dao.create_knowledge_space(request)

    def update_space(self, request: SpaceServeRequest) -> SpaceServeResponse:
        """Update an existing Space entity
        
        Args:
            request (SpaceServeRequest): The request object containing updated space information
        
        Returns:
            SpaceServeResponse: The response object indicating success or failure
        """
        # 获取指定名称或 ID 的空间对象列表
        spaces = self._dao.get_knowledge_space(
            KnowledgeSpaceEntity(id=request.id, name=request.name)
        )
        # 如果未找到对应空间，抛出异常
        if len(spaces) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"no space name named {request.name}",
            )
        # 更新指定空间对象并返回更新结果
        update_obj = self._dao.update_knowledge_space(self._dao.from_request(request))
        return update_obj

    async def create_document(
        self, request: DocumentServeRequest
        ):
        """Create a new document within a specified space
        
        Args:
            request (DocumentServeRequest): The request object containing document information
        
        Returns:
            DocumentServeResponse: The response object indicating success or failure
        """
    ) -> SpaceServeResponse:
        """创建一个新的文档实体

        Args:
            request (KnowledgeSpaceRequest): 请求对象，包含请求信息

        Returns:
            SpaceServeResponse: 响应对象，包含响应信息
        """
        # 根据请求中的 space_id 获取空间信息
        space = self.get({"id": request.space_id})
        # 如果空间信息为 None，则抛出异常，指定的 space_id 未找到
        if space is None:
            raise Exception(f"space id:{request.space_id} not found")
        # 创建查询对象，用于查询文档实体
        query = KnowledgeDocumentEntity(doc_name=request.doc_name, space=space.name)
        # 查询符合条件的文档实体
        documents = self._document_dao.get_knowledge_documents(query)
        # 如果已经存在同名文档，则抛出异常
        if len(documents) > 0:
            raise Exception(f"document name:{request.doc_name} have already named")
        # 如果请求中包含文档文件并且文档类型为 DOCUMENT，则处理上传的文档文件
        if request.doc_file and request.doc_type == KnowledgeType.DOCUMENT.name:
            doc_file = request.doc_file
            # 确保空间对应的上传路径存在，如果不存在则创建
            if not os.path.exists(os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space.name)):
                os.makedirs(os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space.name))
            # 创建临时文件并将上传的文档内容写入其中
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space.name)
            )
            with os.fdopen(tmp_fd, "wb") as tmp:
                tmp.write(await request.doc_file.read())
            # 将临时文件移动到最终的上传路径中
            shutil.move(
                tmp_path,
                os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space.name, doc_file.filename),
            )
            # 设置请求中的文档内容路径
            request.content = os.path.join(
                KNOWLEDGE_UPLOAD_ROOT_PATH, space.name, doc_file.filename
            )
        # 创建文档实体对象
        document = KnowledgeDocumentEntity(
            doc_name=request.doc_name,
            doc_type=request.doc_type,
            space=space.name,
            chunk_size=0,
            status=SyncStatus.TODO.name,
            last_sync=datetime.now(),
            content=request.content,
            result="",
        )
        # 调用 DAO 层方法创建文档实体，获取文档实体的 ID
        doc_id = self._document_dao.create_knowledge_document(document)
        # 如果创建文档失败，则抛出异常
        if doc_id is None:
            raise Exception(f"create document failed, {request.doc_name}")
        # 返回创建的文档实体的 ID
        return doc_id
    # 异步方法，用于同步文档信息
    async def sync_document(self, requests: List[KnowledgeSyncRequest]) -> List:
        """Create a new document entity

        Args:
            request (KnowledgeSpaceRequest): The request

        Returns:
            SpaceServeResponse: The response
        """
        # 初始化空列表用于存储文档 ID
        doc_ids = []
        # 遍历同步请求列表
        for sync_request in requests:
            # 获取空间 ID
            space_id = sync_request.space_id
            # 根据文档 ID 获取文档信息
            docs = self._document_dao.documents_by_ids([sync_request.doc_id])
            # 如果文档列表为空，则抛出异常
            if len(docs) == 0:
                raise Exception(
                    f"there are document called, doc_id: {sync_request.doc_id}"
                )
            # 获取第一个文档
            doc = docs[0]
            # 如果文档状态为运行中或已完成，则抛出异常
            if (
                doc.status == SyncStatus.RUNNING.name
                or doc.status == SyncStatus.FINISHED.name
            ):
                raise Exception(
                    f" doc:{doc.doc_name} status is {doc.status}, can not sync"
                )
            # 获取同步请求的块参数
            chunk_parameters = sync_request.chunk_parameters
            # 如果块策略不是按大小分块，则根据空间上下文设置块大小和重叠
            if chunk_parameters.chunk_strategy != ChunkStrategy.CHUNK_BY_SIZE.name:
                space_context = self.get_space_context(space_id)
                chunk_parameters.chunk_size = (
                    CFG.KNOWLEDGE_CHUNK_SIZE
                    if space_context is None
                    else int(space_context["embedding"]["chunk_size"])
                )
                chunk_parameters.chunk_overlap = (
                    CFG.KNOWLEDGE_CHUNK_OVERLAP
                    if space_context is None
                    else int(space_context["embedding"]["chunk_overlap"])
                )
            # 调用同步文档方法
            await self._sync_knowledge_document(space_id, doc, chunk_parameters)
            # 将文档 ID 添加到列表中
            doc_ids.append(doc.id)
        # 返回文档 ID 列表
        return doc_ids

    # 获取方法，用于获取流实体
    def get(self, request: QUERY_SPEC) -> Optional[SpaceServeResponse]:
        """Get a Flow entity

        Args:
            request (SpaceServeRequest): The request

        Returns:
            SpaceServeResponse: The response
        """
        # TODO: implement your own logic here
        # 从请求构建查询请求
        query_request = request
        # 调用 DAO 的获取方法
        return self._dao.get_one(query_request)

    # 获取文档方法，用于获取文档实体
    def get_document(self, request: QUERY_SPEC) -> Optional[SpaceServeResponse]:
        """Get a Flow entity

        Args:
            request (SpaceServeRequest): The request

        Returns:
            SpaceServeResponse: The response
        """
        # TODO: implement your own logic here
        # 从请求构建查询请求
        query_request = request
        # 调用文档 DAO 的获取方法
        return self._document_dao.get_one(query_request)
    def delete(self, space_id: str) -> Optional[SpaceServeResponse]:
        """Delete a Flow entity

        Args:
            space_id (str): The ID of the space to delete

        Returns:
            Optional[SpaceServeResponse]: The data after deletion, if successful
        """

        # TODO: implement your own logic here

        # 构建查询请求，使用传入的空间ID
        query_request = {"id": space_id}
        
        # 通过空间ID获取空间信息
        space = self.get(query_request)
        if space is None:
            raise HTTPException(status_code=400, detail=f"Space {space_id} not found")
        
        # 准备配置对象，用于连接到向量存储
        config = VectorStoreConfig(
            name=space.name, llm_client=self.llm_client, model_name=None
        )
        
        # 创建向量存储连接器对象，用于删除空间相关的向量
        vector_store_connector = VectorStoreConnector(
            vector_store_type=space.vector_type, vector_store_config=config
        )
        
        # 删除空间对应的向量
        vector_store_connector.delete_vector_name(space.name)
        
        # 准备知识文档实体对象，用于删除空间内的文档块
        document_query = KnowledgeDocumentEntity(space=space.name)
        
        # 获取空间内的所有文档
        documents = self._document_dao.get_documents(document_query)
        
        # 遍历并删除每个文档的文档块
        for document in documents:
            self._chunk_dao.raw_delete(document.id)
        
        # 删除空间内的所有文档
        self._document_dao.raw_delete(document_query)
        
        # 删除空间本身
        self._dao.delete(query_request)
        
        # 返回被删除的空间信息
        return space


    def delete_document(self, document_id: str) -> Optional[DocumentServeResponse]:
        """Delete a Flow entity

        Args:
            document_id (str): The ID of the document to delete

        Returns:
            Optional[DocumentServeResponse]: The data after deletion, if successful
        """

        # 构建查询请求，使用传入的文档ID
        query_request = {"id": document_id}
        
        # 通过文档ID获取文档信息
        document = self._document_dao.get_one(query_request)
        if document is None:
            raise Exception(f"There are no or more than one document with ID {document_id}")
        
        # 根据文档所属空间名称获取空间信息
        spaces = self._dao.get_knowledge_space(
            KnowledgeSpaceEntity(name=document.space)
        )
        
        # 确保仅有一个符合条件的空间存在
        if len(spaces) != 1:
            raise Exception(f"Invalid space name: {document.space}")
        
        space = spaces[0]
        
        # 获取文档关联的向量ID列表
        vector_ids = document.vector_ids
        
        # 如果有向量ID列表，则准备配置对象和连接器，用于删除相关向量
        if vector_ids is not None:
            config = VectorStoreConfig(
                name=space.name, llm_client=self.llm_client, model_name=None
            )
            vector_store_connector = VectorStoreConnector(
                vector_store_type=space.vector_type, vector_store_config=config
            )
            vector_store_connector.delete_by_ids(vector_ids)
        
        # 删除文档块
        self._chunk_dao.raw_delete(document.id)
        
        # 删除文档
        self._document_dao.raw_delete(document)
        
        # 返回被删除的文档信息
        return document
    # 获取 Flow 实体列表的方法，返回一个 SpaceServeResponse 类型的列表

    def get_list(self, request: SpaceServeRequest) -> List[SpaceServeResponse]:
        """Get a list of Flow entities

        Args:
            request (SpaceServeRequest): The request

        Returns:
            List[SpaceServeResponse]: The response
        """
        # TODO: implement your own logic here
        # 构建查询请求对象
        query_request = request
        # 调用数据访问对象（DAO）的方法，获取列表数据并返回
        return self.dao.get_list(query_request)

    # 按页获取 Flow 实体列表的方法，返回一个 PaginationResult 类型的 SpaceServeResponse 列表

    def get_list_by_page(
        self, request: QUERY_SPEC, page: int, page_size: int
    ) -> PaginationResult[SpaceServeResponse]:
        """Get a list of Flow entities by page

        Args:
            request (SpaceServeRequest): The request
            page (int): The page number
            page_size (int): The page size

        Returns:
            List[SpaceServeResponse]: The response
        """
        # 调用数据访问对象（DAO）的方法，按页获取列表数据并返回
        return self.dao.get_list_page(request, page, page_size)

    # 按页获取文档实体列表的方法，返回一个 PaginationResult 类型的 DocumentServeResponse 列表

    def get_document_list(
        self, request: QUERY_SPEC, page: int, page_size: int
    ) -> PaginationResult[DocumentServeResponse]:
        """Get a list of Flow entities by page

        Args:
            request (SpaceServeRequest): The request
            page (int): The page number
            page_size (int): The page size

        Returns:
            List[SpaceServeResponse]: The response
        """
        # 调用私有方法 _document_dao 的方法，按页获取文档列表数据并返回
        return self._document_dao.get_list_page(request, page, page_size)

    # 异步批量文档同步方法，接收 space_id 和 KnowledgeSyncRequest 类型的同步请求列表作为参数
    ) -> List[int]:
        """批量同步知识文档块到向量存储空间
        Args:
            - space: 知识空间名称
            - sync_requests: List[KnowledgeSyncRequest] 同步请求列表
        Returns:
            - List[int]: 文档 IDs 列表
        """
        doc_ids = []  # 初始化文档 IDs 列表

        # 遍历每个同步请求
        for sync_request in sync_requests:
            # 根据 sync_request 中的 doc_id 查询文档信息
            docs = self._document_dao.documents_by_ids([sync_request.doc_id])
            
            # 如果未找到对应文档，抛出异常
            if len(docs) == 0:
                raise Exception(
                    f"未找到名为 doc_id: {sync_request.doc_id} 的文档"
                )
            
            doc = docs[0]  # 获取查询到的文档信息
           
            # 检查文档状态，如果状态为运行中或已完成，抛出异常
            if (
                doc.status == SyncStatus.RUNNING.name
                or doc.status == SyncStatus.FINISHED.name
            ):
                raise Exception(
                    f"文档:{doc.doc_name} 状态为 {doc.status}，无法进行同步操作"
                )
            
            chunk_parameters = sync_request.chunk_parameters  # 获取同步请求的块参数
            
            # 如果块策略不是按大小划分，则根据空间上下文设置块大小和重叠
            if chunk_parameters.chunk_strategy != ChunkStrategy.CHUNK_BY_SIZE.name:
                space_context = self.get_space_context(space_id)
                chunk_parameters.chunk_size = (
                    CFG.KNOWLEDGE_CHUNK_SIZE
                    if space_context is None
                    else int(space_context["embedding"]["chunk_size"])
                )
                chunk_parameters.chunk_overlap = (
                    CFG.KNOWLEDGE_CHUNK_OVERLAP
                    if space_context is None
                    else int(space_context["embedding"]["chunk_overlap"])
                )
            
            # 异步执行知识文档同步操作
            await self._sync_knowledge_document(space_id, doc, chunk_parameters)
            
            doc_ids.append(doc.id)  # 将已同步文档的 ID 添加到结果列表中
        
        return doc_ids  # 返回已同步文档的 IDs 列表
    ) -> None:
        """将知识文档块同步到向量存储"""
        # 获取嵌入工厂组件
        embedding_factory = CFG.SYSTEM_APP.get_component(
            "embedding_factory", EmbeddingFactory
        )
        # 创建嵌入函数
        embedding_fn = embedding_factory.create(
            model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
        )
        # 导入向量存储配置
        from dbgpt.storage.vector_store.base import VectorStoreConfig
        
        # 根据文档值对象创建知识文档实体
        doc = KnowledgeDocumentEntity.from_document_vo(doc_vo)
        
        # 获取指定ID的空间
        space = self.get({"id": space_id})
        
        # 配置向量存储
        config = VectorStoreConfig(
            name=space.name,
            embedding_fn=embedding_fn,
            max_chunks_once_load=CFG.KNOWLEDGE_MAX_CHUNKS_ONCE_LOAD,
            llm_client=self.llm_client,
            model_name=None,
        )
        
        # 创建向量存储连接器
        vector_store_connector = VectorStoreConnector(
            vector_store_type=space.vector_type, vector_store_config=config
        )
        
        knowledge = None
        
        # 如果空间域类型不存在或为正常业务字段类型
        if not space.domain_type or (
            space.domain_type == BusinessFieldType.NORMAL.value
        ):
            # 创建知识对象
            knowledge = KnowledgeFactory.create(
                datasource=doc.content,
                knowledge_type=KnowledgeType.get_by_value(doc.doc_type),
            )
        
        # 设置文档状态为运行中
        doc.status = SyncStatus.RUNNING.name
        
        # 更新文档的修改时间
        doc.gmt_modified = datetime.now()
        
        # 更新知识文档
        self._document_dao.update_knowledge_document(doc)
        
        # 异步创建任务：文档嵌入
        asyncio.create_task(
            self.async_doc_embedding(
                knowledge, chunk_parameters, vector_store_connector, doc, space
            )
        )
        
        # 记录日志：开始保存文档块，文档名称为{doc.doc_name}
        logger.info(f"begin save document chunks, doc:{doc.doc_name}")

    @trace("async_doc_embedding")
    async def async_doc_embedding(
        self, knowledge, chunk_parameters, vector_store_connector, doc, space
    ):
        """异步文档嵌入"""
        # 在这里进行文档嵌入的异步处理

    def get_space_context(self, space_id):
        """获取空间上下文"""
        # 获取指定ID的空间
        space = self.get({"id": space_id})
        
        # 如果空间不存在，则抛出异常
        if space is None:
            raise Exception(
                f"have not found {space_id} space or found more than one space called {space_id}"
            )
        
        # 如果空间上下文不为空，则返回其JSON解析后的结果
        if space.context is not None:
            return json.loads(space.context)
        
        # 如果空间上下文为空，则返回None
        return None
```