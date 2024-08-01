# `.\DB-GPT-src\dbgpt\app\knowledge\service.py`

```py
# 导入需要的模块：json、logging、re、datetime
import json
import logging
import re
from datetime import datetime

# 导入配置模块和数据库访问相关模块
from dbgpt._private.config import Config
from dbgpt.app.knowledge.chunk_db import DocumentChunkDao, DocumentChunkEntity
from dbgpt.app.knowledge.document_db import (
    KnowledgeDocumentDao,
    KnowledgeDocumentEntity,
)
# 导入知识请求相关模块
from dbgpt.app.knowledge.request.request import (
    ChunkQueryRequest,
    DocumentQueryRequest,
    DocumentSummaryRequest,
    KnowledgeDocumentRequest,
    KnowledgeSpaceRequest,
    SpaceArgumentRequest,
)
# 导入知识请求响应相关模块
from dbgpt.app.knowledge.request.response import (
    ChunkQueryResponse,
    DocumentQueryResponse,
    SpaceQueryResponse,
)
# 导入组件类型枚举
from dbgpt.component import ComponentType
# 导入配置中的领域类型常量
from dbgpt.configs import DOMAIN_TYPE_FINANCIAL_REPORT
# 导入嵌入模型配置
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG
# 导入LLM客户端相关模块
from dbgpt.core import LLMClient
from dbgpt.model import DefaultLLMClient
# 导入工作管理器工厂相关模块
from dbgpt.model.cluster import WorkerManagerFactory
# 导入摘要组装器模块
from dbgpt.rag.assembler.summary import SummaryAssembler
# 导入分块参数模块
from dbgpt.rag.chunk_manager import ChunkParameters
# 导入嵌入工厂模块
from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory
# 导入知识类型模块
from dbgpt.rag.knowledge.base import KnowledgeType
# 导入知识工厂模块
from dbgpt.rag.knowledge.factory import KnowledgeFactory
# 导入向量存储连接器模块
from dbgpt.serve.rag.connector import VectorStoreConnector
# 导入知识空间数据访问对象及实体模块
from dbgpt.serve.rag.models.models import KnowledgeSpaceDao, KnowledgeSpaceEntity
# 导入同步状态枚举
from dbgpt.serve.rag.service.service import SyncStatus
# 导入向量存储配置模块
from dbgpt.storage.vector_store.base import VectorStoreConfig
# 导入执行器工厂模块及异步转换函数
from dbgpt.util.executor_utils import ExecutorFactory, blocking_func_to_async
# 导入跟踪器模块及根跟踪器对象
from dbgpt.util.tracer import root_tracer, trace

# 创建知识空间数据访问对象
knowledge_space_dao = KnowledgeSpaceDao()
# 创建知识文档数据访问对象
knowledge_document_dao = KnowledgeDocumentDao()
# 创建文档分块数据访问对象
document_chunk_dao = DocumentChunkDao()

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)
# 获取配置对象
CFG = Config()

# 默认摘要最大迭代调用次数
DEFAULT_SUMMARY_MAX_ITERATION = 5
# 默认摘要并发调用限制
DEFAULT_SUMMARY_CONCURRENCY_LIMIT = 3

# 知识服务类，管理知识空间、知识文档和嵌入管理
class KnowledgeService:
    """KnowledgeService
    知识管理服务：
        -知识空间管理
        -知识文档管理
        -嵌入管理
    """

    def __init__(self):
        pass

    # 获取LLM客户端的属性方法
    @property
    def llm_client(self) -> LLMClient:
        # 获取系统应用的组件管理器工厂，创建工作管理器对象
        worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        # 返回默认LLM客户端对象
        return DefaultLLMClient(worker_manager, True)
    def create_knowledge_space(self, request: KnowledgeSpaceRequest):
        """创建知识空间
        Args:
           - request: KnowledgeSpaceRequest，包含请求的知识空间信息
        """
        # 创建知识空间实体对象
        query = KnowledgeSpaceEntity(
            name=request.name,
        )
        # 如果请求的向量类型为"VectorStore"，则替换为配置中的向量存储类型
        if request.vector_type == "VectorStore":
            request.vector_type = CFG.VECTOR_STORE_TYPE
        # 如果请求的向量类型为"KnowledgeGraph"，则验证知识空间名称的合法性
        if request.vector_type == "KnowledgeGraph":
            knowledge_space_name_pattern = r"^[a-zA-Z0-9\u4e00-\u9fa5]+$"
            if not re.match(knowledge_space_name_pattern, request.name):
                raise Exception(f"space name:{request.name} invalid")
        # 查询符合条件的知识空间
        spaces = knowledge_space_dao.get_knowledge_space(query)
        # 如果存在同名的知识空间，抛出异常
        if len(spaces) > 0:
            raise Exception(f"space name:{request.name} have already named")
        # 创建知识空间并返回其 ID
        space_id = knowledge_space_dao.create_knowledge_space(request)
        return space_id

    def create_knowledge_document(self, space, request: KnowledgeDocumentRequest):
        """创建知识文档
        Args:
           - space: 知识空间对象，指定将要创建文档的空间
           - request: KnowledgeDocumentRequest，包含请求的知识文档信息
        """
        # 查询指定空间中是否已存在同名的文档
        query = KnowledgeDocumentEntity(doc_name=request.doc_name, space=space)
        documents = knowledge_document_dao.get_knowledge_documents(query)
        # 如果存在同名的文档，抛出异常
        if len(documents) > 0:
            raise Exception(f"document name:{request.doc_name} have already named")
        # 创建知识文档实体对象
        document = KnowledgeDocumentEntity(
            doc_name=request.doc_name,
            doc_type=request.doc_type,
            space=space,
            chunk_size=0,
            status=SyncStatus.TODO.name,
            last_sync=datetime.now(),
            content=request.content,
            result="",
        )
        # 创建知识文档并返回其 ID
        doc_id = knowledge_document_dao.create_knowledge_document(document)
        # 如果创建失败，抛出异常
        if doc_id is None:
            raise Exception(f"create document failed, {request.doc_name}")
        return doc_id

    def get_knowledge_space(self, request: KnowledgeSpaceRequest):
        """获取知识空间列表及相关信息
        Args:
           - request: KnowledgeSpaceRequest，包含请求的知识空间信息
        """
        # 查询符合条件的知识空间
        query = KnowledgeSpaceEntity(
            name=request.name, vector_type=request.vector_type, owner=request.owner
        )
        spaces = knowledge_space_dao.get_knowledge_space(query)
        # 查询每个知识空间中的文档数量
        space_names = [space.name for space in spaces]
        docs_count = knowledge_document_dao.get_knowledge_documents_count_bulk(
            space_names
        )
        # 构建返回的知识空间响应列表
        responses = []
        for space in spaces:
            res = SpaceQueryResponse()
            res.id = space.id
            res.name = space.name
            res.vector_type = space.vector_type
            res.domain_type = space.domain_type
            res.desc = space.desc
            res.owner = space.owner
            res.gmt_created = space.gmt_created
            res.gmt_modified = space.gmt_modified
            res.context = space.context
            res.docs = docs_count.get(space.name, 0)
            responses.append(res)
        return responses
    # 显示知识空间的参数
    def arguments(self, space_name):
        """show knowledge space arguments
        Args:
            - space_name: Knowledge Space Name
        """
        # 创建查询对象，用于获取指定名称的知识空间实体
        query = KnowledgeSpaceEntity(name=space_name)
        # 调用数据访问层方法，获取符合查询条件的知识空间列表
        spaces = knowledge_space_dao.get_knowledge_space(query)
        # 如果返回的空间列表不唯一，则抛出异常
        if len(spaces) != 1:
            raise Exception(f"there are no or more than one space called {space_name}")
        # 获取第一个匹配的知识空间对象
        space = spaces[0]
        # 如果空间的上下文信息为空，使用默认上下文信息进行构建
        if space.context is None:
            context = self._build_default_context()
        else:
            context = space.context
        # 将上下文信息从 JSON 字符串转换为 Python 对象并返回
        return json.loads(context)

    # 保存参数
    def argument_save(self, space_name, argument_request: SpaceArgumentRequest):
        """save argument
        Args:
            - space_name: Knowledge Space Name
            - argument_request: SpaceArgumentRequest
        """
        # 创建查询对象，用于获取指定名称的知识空间实体
        query = KnowledgeSpaceEntity(name=space_name)
        # 调用数据访问层方法，获取符合查询条件的知识空间列表
        spaces = knowledge_space_dao.get_knowledge_space(query)
        # 如果返回的空间列表不唯一，则抛出异常
        if len(spaces) != 1:
            raise Exception(f"there are no or more than one space called {space_name}")
        # 获取第一个匹配的知识空间对象
        space = spaces[0]
        # 将传入的参数保存到空间的上下文信息中
        space.context = argument_request.argument
        # 调用数据访问层方法，更新知识空间对象
        return knowledge_space_dao.update_knowledge_space(space)

    # 获取知识文档
    def get_knowledge_documents(self, space, request: DocumentQueryRequest):
        """get knowledge documents
        Args:
            - space: Knowledge Space Name
            - request: DocumentQueryRequest
        Returns:
            - res DocumentQueryResponse
        """
        # 初始化总数为 None
        total = None
        # 获取请求中的页码
        page = request.page
        # 如果请求中包含文档 ID 并且数量大于 0，则根据文档 ID 查询文档数据
        if request.doc_ids and len(request.doc_ids) > 0:
            data = knowledge_document_dao.documents_by_ids(request.doc_ids)
        else:
            # 否则，创建查询对象，根据指定条件从数据库中获取知识文档数据
            query = KnowledgeDocumentEntity(
                doc_name=request.doc_name,
                doc_type=request.doc_type,
                space=space,
                status=request.status,
            )
            # 调用数据访问层方法，获取符合条件的知识文档数据
            data = knowledge_document_dao.get_knowledge_documents(
                query, page=request.page, page_size=request.page_size
            )
            # 获取符合条件的知识文档总数
            total = knowledge_document_dao.get_knowledge_documents_count(query)
        # 返回包含文档数据、总数和页码的文档查询响应对象
        return DocumentQueryResponse(data=data, total=total, page=page)
    async def document_summary(self, request: DocumentSummaryRequest):
        """获取文档摘要
        Args:
            - request: DocumentSummaryRequest，文档摘要请求对象
        """
        # 创建知识文档实体对象，使用请求中的文档ID
        doc_query = KnowledgeDocumentEntity(id=request.doc_id)
        # 从数据库中获取符合条件的知识文档列表
        documents = knowledge_document_dao.get_documents(doc_query)
        # 如果返回的文档列表长度不为1，则抛出异常，表示未找到唯一匹配的文档
        if len(documents) != 1:
            raise Exception(f"can not found document for {request.doc_id}")
        # 获取匹配到的文档对象
        document = documents[0]
        
        # 导入 WorkerManagerFactory 类
        from dbgpt.model.cluster import WorkerManagerFactory
        # 从系统应用配置中获取 WorkerManagerFactory 组件实例并创建 WorkerManager 对象
        worker_manager = CFG.SYSTEM_APP.get_component(
            ComponentType.WORKER_MANAGER_FACTORY, WorkerManagerFactory
        ).create()
        
        # 创建文档分块参数对象，定义分块策略、大小和重叠区域
        chunk_parameters = ChunkParameters(
            chunk_strategy="CHUNK_BY_SIZE",
            chunk_size=CFG.KNOWLEDGE_CHUNK_SIZE,
            chunk_overlap=CFG.KNOWLEDGE_CHUNK_OVERLAP,
        )
        
        # 查询指定文档的分块实体列表
        chunk_entities = document_chunk_dao.get_document_chunks(
            DocumentChunkEntity(document_id=document.id)
        )
        
        # 如果文档状态不是运行中或分块实体列表为空，则执行同步操作
        if (
            document.status not in [SyncStatus.RUNNING.name]
            and len(chunk_entities) == 0
        ):
            # 导入 Service 类
            from dbgpt.serve.rag.service.service import Service
            # 获取 RAG 服务的单例实例
            rag_service = Service.get_instance(CFG.SYSTEM_APP)
            # 根据文档所属空间名称获取空间对象
            space = rag_service.get({"name": document.space})
            # 将文档实体转换为文档值对象列表
            document_vo = KnowledgeDocumentEntity.to_document_vo(documents)
            # 异步执行知识文档同步操作
            await rag_service._sync_knowledge_document(
                space_id=space.id,
                doc_vo=document_vo[0],
                chunk_parameters=chunk_parameters,
            )
        
        # 根据文档内容和类型创建知识对象
        knowledge = KnowledgeFactory.create(
            datasource=document.content,
            knowledge_type=KnowledgeType.get_by_value(document.doc_type),
        )
        
        # 创建摘要组装器对象，配置知识对象、模型名称、LLM 客户端等参数
        assembler = SummaryAssembler(
            knowledge=knowledge,
            model_name=request.model_name,
            llm_client=DefaultLLMClient(
                worker_manager=worker_manager, auto_convert_message=True
            ),
            language=CFG.LANGUAGE,
            chunk_parameters=chunk_parameters,
        )
        
        # 异步生成摘要内容
        summary = await assembler.generate_summary()

        # 如果摘要组装器中的分块列表为空，则抛出异常，表示未找到对应文档的分块
        if len(assembler.get_chunks()) == 0:
            raise Exception(f"can not found chunks for {request.doc_id}")

        # 调用私有方法，提取摘要并返回结果
        return await self._llm_extract_summary(
            summary, request.conv_uid, request.model_name
        )

    def update_knowledge_space(
        self, space_id: int, space_request: KnowledgeSpaceRequest
    ):
        """更新知识空间
        Args:
            - space_id: 空间ID
            - space_request: KnowledgeSpaceRequest，知识空间请求对象
        """
        # 创建知识空间实体对象，使用给定的空间ID和请求参数
        entity = KnowledgeSpaceEntity(
            id=space_id,
            name=space_request.name,
            vector_type=space_request.vector_type,
            desc=space_request.desc,
            owner=space_request.owner,
        )

        # 调用数据访问对象更新知识空间信息
        knowledge_space_dao.update_knowledge_space(entity)
    def delete_space(self, space_name: str):
        """删除知识空间

        Args:
            - space_name: 知识空间名称
        """

        # 通过知识空间名称从数据库获取知识空间对象列表
        spaces = knowledge_space_dao.get_knowledge_space(
            KnowledgeSpaceEntity(name=space_name)
        )

        # 如果获取到的知识空间对象数量不为1，则抛出异常
        if len(spaces) != 1:
            raise Exception(f"invalid space name:{space_name}")

        # 获取第一个知识空间对象
        space = spaces[0]

        # 获取嵌入工厂组件并创建嵌入函数
        embedding_factory = CFG.SYSTEM_APP.get_component(
            "embedding_factory", EmbeddingFactory
        )
        embedding_fn = embedding_factory.create(
            model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
        )

        # 创建向量存储配置对象
        config = VectorStoreConfig(
            name=space.name,
            embedding_fn=embedding_fn,
            llm_client=self.llm_client,
            model_name=None,
        )

        # 如果知识空间的域类型为财务报告，则使用本地数据库管理器删除相关数据库
        if space.domain_type == DOMAIN_TYPE_FINANCIAL_REPORT:
            conn_manager = CFG.local_db_manager
            conn_manager.delete_db(f"{space.name}_fin_report")

        # 创建向量存储连接器对象
        vector_store_connector = VectorStoreConnector(
            vector_store_type=space.vector_type, vector_store_config=config
        )

        # 删除向量存储中的向量
        vector_store_connector.delete_vector_name(space.name)

        # 构建知识文档实体查询对象
        document_query = KnowledgeDocumentEntity(space=space.name)

        # 获取指定知识空间下的所有文档
        documents = knowledge_document_dao.get_documents(document_query)

        # 遍历所有文档，逐个删除其文档块
        for document in documents:
            document_chunk_dao.raw_delete(document.id)

        # 删除知识文档
        knowledge_document_dao.raw_delete(document_query)

        # 删除知识空间
        return knowledge_space_dao.delete_knowledge_space(space)
    def delete_document(self, space_name: str, doc_name: str):
        """delete document method
        
        Args:
            - space_name: knowledge space name
            - doc_name: document name
        """
        # 创建知识文档实体对象，用于查询
        document_query = KnowledgeDocumentEntity(doc_name=doc_name, space=space_name)
    # 异步将文档嵌入到向量数据库中
    def async_doc_embedding(self, assembler, chunk_docs, doc):
        """async document embedding into vector db
        Args:
            - client: EmbeddingEngine Client
            - chunk_docs: List[Document]
            - doc: KnowledgeDocumentEntity
        """

        # 记录日志，显示文档名称和文档分块的长度
        logger.info(
            f"async doc embedding sync, doc:{doc.doc_name}, chunks length is {len(chunk_docs)}"
        )
        try:
            # 开始一个新的跟踪 span
            with root_tracer.start_span(
                "app.knowledge.assembler.persist",
                metadata={"doc": doc.doc_name, "chunks": len(chunk_docs)},
            ):
                # 持久化向量并获取向量 ID
                vector_ids = assembler.persist()
            # 设置文档状态为完成
            doc.status = SyncStatus.FINISHED.name
            doc.result = "document embedding success"
            # 如果有向量 ID，则将其转换为字符串并保存到文档中
            if vector_ids is not None:
                doc.vector_ids = ",".join(vector_ids)
            logger.info(f"async document embedding, success:{doc.doc_name}")
            # 保存文档分块的详细信息
            chunk_entities = [
                DocumentChunkEntity(
                    doc_name=doc.doc_name,
                    doc_type=doc.doc_type,
                    document_id=doc.id,
                    content=chunk_doc.content,
                    meta_info=str(chunk_doc.metadata),
                    gmt_created=datetime.now(),
                    gmt_modified=datetime.now(),
                )
                for chunk_doc in chunk_docs
            ]
            document_chunk_dao.create_documents_chunks(chunk_entities)
        except Exception as e:
            # 如果出现异常，设置文档状态为失败，并记录异常信息
            doc.status = SyncStatus.FAILED.name
            doc.result = "document embedding failed" + str(e)
            logger.error(f"document embedding, failed:{doc.doc_name}, {str(e)}")
        # 更新知识文档并返回
        return knowledge_document_dao.update_knowledge_document(doc)

    # 构建默认上下文
    def _build_default_context(self):
        from dbgpt.app.scene.chat_knowledge.v1.prompt import (
            _DEFAULT_TEMPLATE,
            PROMPT_SCENE_DEFINE,
        )

        # 定义上下文模板
        context_template = {
            "embedding": {
                "topk": CFG.KNOWLEDGE_SEARCH_TOP_SIZE,
                "recall_score": CFG.KNOWLEDGE_SEARCH_RECALL_SCORE,
                "recall_type": "TopK",
                "model": EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL].rsplit("/", 1)[-1],
                "chunk_size": CFG.KNOWLEDGE_CHUNK_SIZE,
                "chunk_overlap": CFG.KNOWLEDGE_CHUNK_OVERLAP,
            },
            "prompt": {
                "max_token": 2000,
                "scene": PROMPT_SCENE_DEFINE,
                "template": _DEFAULT_TEMPLATE,
            },
            "summary": {
                "max_iteration": DEFAULT_SUMMARY_MAX_ITERATION,
                "concurrency_limit": DEFAULT_SUMMARY_CONCURRENCY_LIMIT,
            },
        }
        # 将上下文模板转换为字符串并返回
        context_template_string = json.dumps(context_template, indent=4)
        return context_template_string
    def get_space_context(self, space_name):
        """
        获取空间上下文信息
        Args:
           - space_name: 空间名称
        """
        # 创建一个 KnowledgeSpaceRequest 对象
        request = KnowledgeSpaceRequest()
        # 设置请求对象的名称属性为给定的 space_name
        request.name = space_name
        # 调用 get_knowledge_space 方法获取符合条件的知识空间列表
        spaces = self.get_knowledge_space(request)
        # 如果返回的空间列表不唯一，抛出异常
        if len(spaces) != 1:
            raise Exception(
                f"未找到名为 {space_name} 的空间，或者找到多个名为 {space_name} 的空间"
            )
        # 获取第一个符合条件的空间对象
        space = spaces[0]
        # 如果空间对象的上下文属性不为空，解析 JSON 格式的上下文信息并返回
        if space.context is not None:
            return json.loads(spaces[0].context)
        # 如果空间对象的上下文属性为空，返回 None
        return None

    async def _llm_extract_summary(
        self, doc: str, conn_uid: str, model_name: str = None
    ):
        """
        从文本中提取三元组（使用 LLM 模型）
        Args:
            doc: 文档内容
            conn_uid: str，聊天会话 ID
            model_name: str，模型名称
        Returns:
            chat: BaseChat，精炼后的摘要信息对象
        """
        # 导入 ChatScene 模块中的 Chat 类
        from dbgpt.app.scene import ChatScene

        # 设置聊天参数
        chat_param = {
            "chat_session_id": conn_uid,
            "current_user_input": "",
            "select_param": doc,
            "model_name": model_name,
            "model_cache_enable": False,
        }
        # 获取系统默认执行器对象
        executor = CFG.SYSTEM_APP.get_component(
            ComponentType.EXECUTOR_DEFAULT, ExecutorFactory
        ).create()
        # 导入 CHAT_FACTORY
        from dbgpt.app.openapi.api_v1.api_v1 import CHAT_FACTORY

        # 使用异步方式调用 blocking_func_to_async 函数执行 ChatScene 中的 ExtractRefineSummary 操作
        chat = await blocking_func_to_async(
            executor,
            CHAT_FACTORY.get_implementation,
            ChatScene.ExtractRefineSummary.value(),
            **{"chat_param": chat_param},
        )
        # 返回 chat 对象作为摘要信息的结果
        return chat
    # 定义一个方法，用于查询指定知识空间中的图数据
    def query_graph(self, space_name, limit):
        # 获取嵌入工厂组件实例
        embedding_factory = CFG.SYSTEM_APP.get_component(
            "embedding_factory", EmbeddingFactory
        )
        # 根据配置中的嵌入模型名称创建对应的嵌入函数
        embedding_fn = embedding_factory.create(
            model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
        )
        # 获取指定名称的知识空间的详细信息
        spaces = self.get_knowledge_space(KnowledgeSpaceRequest(name=space_name))
        # 如果获取到的空间信息不唯一，抛出异常
        if len(spaces) != 1:
            raise Exception(f"invalid space name:{space_name}")
        # 从获取到的空间信息中取第一个空间作为当前操作的空间
        space = spaces[0]
        # 打印当前配置中的LLM模型
        print(CFG.LLM_MODEL)
        # 创建向量存储配置对象，用于向量存储连接器的初始化
        config = VectorStoreConfig(
            name=space.name,
            embedding_fn=embedding_fn,
            max_chunks_once_load=CFG.KNOWLEDGE_MAX_CHUNKS_ONCE_LOAD,
            llm_client=self.llm_client,
            model_name=None,  # 模型名称暂未指定
        )

        # 创建向量存储连接器，用于实际执行图查询操作
        vector_store_connector = VectorStoreConnector(
            vector_store_type=space.vector_type, vector_store_config=config
        )
        # 调用向量存储连接器的方法执行图查询操作，获取查询结果图
        graph = vector_store_connector.client.query_graph(limit=limit)
        
        # 初始化返回结果字典
        res = {"nodes": [], "edges": []}
        
        # 遍历图中的所有顶点，将顶点信息添加到结果字典中的节点列表中
        for node in graph.vertices():
            res["nodes"].append({"vid": node.vid})
        
        # 遍历图中的所有边，将边信息添加到结果字典中的边列表中
        for edge in graph.edges():
            res["edges"].append(
                {
                    "src": edge.sid,
                    "dst": edge.tid,
                    "label": edge.props[graph.edge_label],  # 获取边的标签信息
                }
            )
        
        # 返回最终的查询结果字典
        return res
```