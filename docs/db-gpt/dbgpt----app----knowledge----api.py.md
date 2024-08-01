# `.\DB-GPT-src\dbgpt\app\knowledge\api.py`

```py
# 导入所需模块
import logging  # 导入日志模块
import os  # 导入操作系统模块
import shutil  # 导入文件操作模块
import tempfile  # 导入临时文件模块
from typing import List  # 导入类型注解模块

from fastapi import APIRouter, Depends, File, Form, UploadFile  # 导入FastAPI相关模块

# 导入配置相关模块和类
from dbgpt._private.config import Config  
# 导入请求相关的类
from dbgpt.app.knowledge.request.request import (
    ChunkQueryRequest,
    DocumentQueryRequest,
    DocumentSummaryRequest,
    DocumentSyncRequest,
    EntityExtractRequest,
    GraphVisRequest,
    KnowledgeDocumentRequest,
    KnowledgeQueryRequest,
    KnowledgeSpaceRequest,
    SpaceArgumentRequest,
)
# 导入请求响应相关的类
from dbgpt.app.knowledge.request.response import KnowledgeQueryResponse  
# 导入知识服务模块
from dbgpt.app.knowledge.service import KnowledgeService  
# 导入API v1相关模块和类
from dbgpt.app.openapi.api_v1.api_v1 import no_stream_generator, stream_generator  
# 导入结果相关的类
from dbgpt.app.openapi.api_view_model import Result  
# 导入配置的标签键
from dbgpt.configs import TAG_KEY_KNOWLEDGE_FACTORY_DOMAIN_TYPE  
# 导入嵌入模型配置和知识上传根路径
from dbgpt.configs.model_config import (
    EMBEDDING_MODEL_CONFIG,
    KNOWLEDGE_UPLOAD_ROOT_PATH,
)
# 导入DAG管理器
from dbgpt.core.awel.dag.dag_manager import DAGManager  
# 导入Chunk参数
from dbgpt.rag import ChunkParameters  
# 导入嵌入工厂
from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory  
# 导入块策略
from dbgpt.rag.knowledge.base import ChunkStrategy  
# 导入知识工厂
from dbgpt.rag.knowledge.factory import KnowledgeFactory  
# 导入嵌入检索器
from dbgpt.rag.retriever.embedding import EmbeddingRetriever  
# 导入知识配置响应类、知识域类型、知识存储类型和知识同步请求
from dbgpt.serve.rag.api.schemas import (
    KnowledgeConfigResponse,
    KnowledgeDomainType,
    KnowledgeStorageType,
    KnowledgeSyncRequest,
)
# 导入向量存储连接器
from dbgpt.serve.rag.connector import VectorStoreConnector  
# 导入服务类
from dbgpt.serve.rag.service.service import Service  
# 导入向量存储配置
from dbgpt.storage.vector_store.base import VectorStoreConfig  
# 导入国际化翻译工具
from dbgpt.util.i18n_utils import _  
# 导入追踪器类型和根追踪器
from dbgpt.util.tracer import SpanType, root_tracer  

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

# 获取配置对象
CFG = Config()

# 创建API路由对象
router = APIRouter()

# 创建知识空间服务实例
knowledge_space_service = KnowledgeService()


# 获取Rag服务的函数
def get_rag_service() -> Service:
    """Get Rag Service."""
    return Service.get_instance(CFG.SYSTEM_APP)


# 获取DAG管理器的函数
def get_dag_manager() -> DAGManager:
    """Get DAG Manager."""
    return DAGManager.get_instance(CFG.SYSTEM_APP)


# 定义API路由的POST方法，用于添加知识空间
@router.post("/knowledge/space/add")
def space_add(request: KnowledgeSpaceRequest):
    """Add a new knowledge space."""
    # 打印请求参数
    print(f"/space/add params: {request}")
    try:
        # 调用知识空间服务创建知识空间
        knowledge_space_service.create_knowledge_space(request)
        # 返回成功结果
        return Result.succ([])
    except Exception as e:
        # 返回失败结果，包含错误代码和消息
        return Result.failed(code="E000X", msg=f"space add error {e}")


# 定义API路由的POST方法，用于列出知识空间
@router.post("/knowledge/space/list")
def space_list(request: KnowledgeSpaceRequest):
    """List all knowledge spaces."""
    # 打印请求参数
    print(f"/space/list params:")
    try:
        # 调用知识空间服务获取知识空间列表，并返回成功结果
        return Result.succ(knowledge_space_service.get_knowledge_space(request))
    except Exception as e:
        # 返回失败结果，包含错误代码和消息
        return Result.failed(code="E000X", msg=f"space list error {e}")


# 定义API路由的POST方法，用于删除知识空间
@router.post("/knowledge/space/delete")
def space_delete(request: KnowledgeSpaceRequest):
    """Delete a knowledge space."""
    # 打印请求参数
    print(f"/space/delete params:")
    try:
        # 调用知识空间服务删除指定名称的知识空间，并返回成功结果
        return Result.succ(knowledge_space_service.delete_space(request.name))
    except Exception as e:
        # 返回失败结果，包含错误代码和消息
        return Result.failed(code="E000X", msg=f"space delete error {e}")
@router.post("/knowledge/{space_name}/arguments")
def arguments(space_name: str):
    # 打印请求参数的路径信息
    print(f"/knowledge/space/arguments params:")
    try:
        # 调用 knowledge_space_service 的 arguments 方法，返回成功的 Result 对象
        return Result.succ(knowledge_space_service.arguments(space_name))
    except Exception as e:
        # 捕获异常并返回失败的 Result 对象，附带错误码和错误信息
        return Result.failed(code="E000X", msg=f"space arguments error {e}")


@router.post("/knowledge/{space_name}/argument/save")
def arguments_save(space_name: str, argument_request: SpaceArgumentRequest):
    # 打印请求参数的路径信息
    print(f"/knowledge/space/argument/save params:")
    try:
        # 调用 knowledge_space_service 的 argument_save 方法，返回成功的 Result 对象
        return Result.succ(
            knowledge_space_service.argument_save(space_name, argument_request)
        )
    except Exception as e:
        # 捕获异常并返回失败的 Result 对象，附带错误码和错误信息
        return Result.failed(code="E000X", msg=f"space save error {e}")


@router.post("/knowledge/{space_name}/document/add")
def document_add(space_name: str, request: KnowledgeDocumentRequest):
    # 打印请求参数的路径信息和请求内容
    print(f"/document/add params: {space_name}, {request}")
    try:
        # 调用 knowledge_space_service 的 create_knowledge_document 方法，返回成功的 Result 对象
        return Result.succ(
            knowledge_space_service.create_knowledge_document(
                space=space_name, request=request
            )
        )
    except Exception as e:
        # 捕获异常并返回失败的 Result 对象，附带错误码和错误信息
        return Result.failed(code="E000X", msg=f"document add error {e}")


@router.get("/knowledge/document/chunkstrategies")
def chunk_strategies():
    """Get chunk strategies"""
    # 打印请求参数的路径信息
    print(f"/document/chunkstrategies:")
    try:
        # 构建包含不同 chunk 策略信息的列表，并返回成功的 Result 对象
        return Result.succ(
            [
                {
                    "strategy": strategy.name,
                    "name": strategy.value[2],
                    "description": strategy.value[3],
                    "parameters": strategy.value[1],
                    "suffix": [
                        knowledge.document_type().value
                        for knowledge in KnowledgeFactory.subclasses()
                        if strategy in knowledge.support_chunk_strategy()
                        and knowledge.document_type() is not None
                    ],
                    "type": set(
                        [
                            knowledge.type().value
                            for knowledge in KnowledgeFactory.subclasses()
                            if strategy in knowledge.support_chunk_strategy()
                        ]
                    ),
                }
                for strategy in ChunkStrategy
            ]
        )
    except Exception as e:
        # 捕获异常并返回失败的 Result 对象，附带错误码和错误信息
        return Result.failed(code="E000X", msg=f"chunk strategies error {e}")


@router.get("/knowledge/space/config", response_model=Result[KnowledgeConfigResponse])
async def space_config() -> Result[KnowledgeConfigResponse]:
    """Get space config"""
    try:
        # 初始化一个空的存储列表，用于保存不同类型的知识存储配置
        storage_list: List[KnowledgeStorageType] = []
        # 获取 DAG 管理器的实例
        dag_manager: DAGManager = get_dag_manager()
        
        # Vector Storage
        # 定义默认的向量存储域类型，并获取与知识工厂域类型标签相关联的 DAGs
        vs_domain_types = [KnowledgeDomainType(name="Normal", desc="Normal")]
        dag_map = dag_manager.get_dags_by_tag_key(TAG_KEY_KNOWLEDGE_FACTORY_DOMAIN_TYPE)
        
        # 遍历 DAG 映射，将每个域类型及其对应的描述添加到向量存储域类型列表中
        for domain_type, dags in dag_map.items():
            vs_domain_types.append(
                KnowledgeDomainType(
                    name=domain_type, desc=dags[0].description or domain_type
                )
            )

        # 向存储列表中添加向量存储类型的配置
        storage_list.append(
            KnowledgeStorageType(
                name="VectorStore",
                desc=_("Vector Store"),
                domain_types=vs_domain_types,
            )
        )
        
        # Graph Storage
        # 向存储列表中添加知识图谱存储类型的配置
        storage_list.append(
            KnowledgeStorageType(
                name="KnowledgeGraph",
                desc=_("Knowledge Graph"),
                domain_types=[KnowledgeDomainType(name="Normal", desc="Normal")],
            )
        )
        
        # Full Text
        # 向存储列表中添加全文存储类型的配置
        storage_list.append(
            KnowledgeStorageType(
                name="FullText",
                desc=_("Full Text"),
                domain_types=[KnowledgeDomainType(name="Normal", desc="Normal")],
            )
        )

        # 成功返回知识配置响应结果，包含存储列表
        return Result.succ(
            KnowledgeConfigResponse(
                storage=storage_list,
            )
        )
    except Exception as e:
        # 处理异常情况，返回失败结果，包含错误代码和消息
        return Result.failed(code="E000X", msg=f"space config error {e}")
# 定义一个 POST 请求处理函数，用于获取指定知识空间下文档列表
@router.post("/knowledge/{space_name}/document/list")
def document_list(space_name: str, query_request: DocumentQueryRequest):
    # 打印请求的参数信息
    print(f"/document/list params: {space_name}, {query_request}")
    try:
        # 调用知识空间服务的方法，获取指定空间下的文档列表，并返回成功的结果
        return Result.succ(
            knowledge_space_service.get_knowledge_documents(space_name, query_request)
        )
    except Exception as e:
        # 如果出现异常，返回失败的结果，包括错误代码和错误消息
        return Result.failed(code="E000X", msg=f"document list error {e}")


# 定义一个 POST 请求处理函数，用于查询指定知识空间的图形可视化
@router.post("/knowledge/{space_name}/graphvis")
def graph_vis(space_name: str, query_request: GraphVisRequest):
    # 打印请求的参数信息
    print(f"/document/list params: {space_name}, {query_request}")
    # 打印查询请求中的限制参数
    print(query_request.limit)
    try:
        # 调用知识空间服务的方法，查询指定空间的图形可视化数据，并返回成功的结果
        return Result.succ(
            knowledge_space_service.query_graph(
                space_name=space_name, limit=query_request.limit
            )
        )
    except Exception as e:
        # 如果出现异常，返回失败的结果，包括错误代码和错误消息
        return Result.failed(code="E000X", msg=f"get graph vis error {e}")


# 定义一个 POST 请求处理函数，用于删除指定知识空间下的文档
@router.post("/knowledge/{space_name}/document/delete")
def document_delete(space_name: str, query_request: DocumentQueryRequest):
    # 打印请求的参数信息
    print(f"/document/list params: {space_name}, {query_request}")
    try:
        # 调用知识空间服务的方法，删除指定空间下指定名称的文档，并返回成功的结果
        return Result.succ(
            knowledge_space_service.delete_document(space_name, query_request.doc_name)
        )
    except Exception as e:
        # 如果出现异常，返回失败的结果，包括错误代码和错误消息
        return Result.failed(code="E000X", msg=f"document delete error {e}")


# 定义一个异步的 POST 请求处理函数，用于上传文档到指定的知识空间
async def document_upload(
    space_name: str,
    doc_name: str = Form(...),
    doc_type: str = Form(...),
    doc_file: UploadFile = File(...),
):
    # 打印上传文档的参数信息
    print(f"/document/upload params: {space_name}")
    try:
        # 检查是否有上传的文档
        if doc_file:
            # 检查目标目录是否存在，若不存在则创建
            if not os.path.exists(os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space_name)):
                os.makedirs(os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space_name))
            
            # 在目标目录中创建临时文件并获取文件描述符和路径
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space_name)
            )
            # 将上传的文档内容写入临时文件
            with os.fdopen(tmp_fd, "wb") as tmp:
                tmp.write(await doc_file.read())
            
            # 将临时文件移动到目标位置，使用原始文件名
            shutil.move(
                tmp_path,
                os.path.join(KNOWLEDGE_UPLOAD_ROOT_PATH, space_name, doc_file.filename),
            )
            
            # 构建知识文档请求对象
            request = KnowledgeDocumentRequest()
            request.doc_name = doc_name
            request.doc_type = doc_type
            request.content = os.path.join(
                KNOWLEDGE_UPLOAD_ROOT_PATH, space_name, doc_file.filename
            )
            
            # 获取知识空间信息
            space_res = knowledge_space_service.get_knowledge_space(
                KnowledgeSpaceRequest(name=space_name)
            )
            
            # 如果知识空间不存在，创建默认空间（除非空间名为"default"）
            if len(space_res) == 0:
                if "default" != space_name:
                    raise Exception(f"you have not create your knowledge space.")
                knowledge_space_service.create_knowledge_space(
                    KnowledgeSpaceRequest(
                        name=space_name,
                        desc="first db-gpt rag application",
                        owner="dbgpt",
                    )
                )
            
            # 成功时返回成功的结果，调用创建知识文档的服务方法
            return Result.succ(
                knowledge_space_service.create_knowledge_document(
                    space=space_name, request=request
                )
            )
        
        # 如果没有上传文档，则返回失败信息
        return Result.failed(code="E000X", msg=f"doc_file is None")
    
    # 捕获所有异常，并返回失败信息
    except Exception as e:
        return Result.failed(code="E000X", msg=f"document add error {e}")
# 定义一个异步 POST 路由处理函数，用于文档同步操作
@router.post("/knowledge/{space_name}/document/sync")
async def document_sync(
    space_name: str,  # 从 URL 路径中获取空间名称
    request: DocumentSyncRequest,  # 接收 DocumentSyncRequest 类型的请求体
    service: Service = Depends(get_rag_service),  # 使用 get_rag_service 依赖注入 Service 对象
):
    logger.info(f"Received params: {space_name}, {request}")  # 记录接收到的参数信息
    try:
        space = service.get({"name": space_name})  # 根据空间名称从 service 中获取对应的空间对象
        if space is None:
            return Result.failed(code="E000X", msg=f"space {space_name} not exist")  # 若空间对象不存在则返回失败结果
        if request.doc_ids is None or len(request.doc_ids) == 0:
            return Result.failed(code="E000X", msg="doc_ids is None")  # 若请求中的 doc_ids 为空则返回失败结果
        # 创建一个 KnowledgeSyncRequest 对象
        sync_request = KnowledgeSyncRequest(
            doc_id=request.doc_ids[0],  # 使用请求中的第一个 doc_id
            space_id=str(space.id),  # 使用空间对象的 ID，并转换为字符串类型
            model_name=request.model_name,  # 使用请求中的模型名称
        )
        # 设置同步请求的分块参数
        sync_request.chunk_parameters = ChunkParameters(
            chunk_strategy="Automatic",  # 分块策略为自动
            chunk_size=request.chunk_size or 512,  # 分块大小，默认为 512
            chunk_overlap=request.chunk_overlap or 50,  # 分块重叠，默认为 50
        )
        # 调用 service 的异步方法进行文档同步，并获取返回的 doc_ids
        doc_ids = await service.sync_document(requests=[sync_request])
        # 返回成功结果，并携带同步的 doc_ids
        return Result.succ(doc_ids)
    except Exception as e:
        # 捕获异常，并返回失败结果，包含错误码和异常信息
        return Result.failed(code="E000X", msg=f"document sync error {e}")


# 定义一个异步 POST 路由处理函数，用于批量文档同步操作
@router.post("/knowledge/{space_name}/document/sync_batch")
async def batch_document_sync(
    space_name: str,  # 从 URL 路径中获取空间名称
    request: List[KnowledgeSyncRequest],  # 接收 KnowledgeSyncRequest 类型的列表请求体
    service: Service = Depends(get_rag_service),  # 使用 get_rag_service 依赖注入 Service 对象
):
    logger.info(f"Received params: {space_name}, {request}")  # 记录接收到的参数信息
    try:
        space = service.get({"name": space_name})  # 根据空间名称从 service 中获取对应的空间对象
        for sync_request in request:
            sync_request.space_id = space.id  # 设置每个同步请求的 space_id 为当前空间对象的 ID
        # 调用 service 的异步方法进行文档批量同步，并获取返回的 doc_ids
        doc_ids = await service.sync_document(requests=request)
        # 返回成功结果，并携带同步的 doc_ids
        return Result.succ({"tasks": doc_ids})
    except Exception as e:
        # 捕获异常，并返回失败结果，包含错误码和异常信息
        return Result.failed(code="E000X", msg=f"document sync batch error {e}")


# 定义一个同步 POST 路由处理函数，用于列出文档分块信息
@router.post("/knowledge/{space_name}/chunk/list")
def document_list(space_name: str, query_request: ChunkQueryRequest):
    print(f"/document/list params: {space_name}, {query_request}")  # 打印接收到的参数信息
    try:
        # 调用 knowledge_space_service 的方法获取文档分块信息，并返回成功结果
        return Result.succ(knowledge_space_service.get_document_chunks(query_request))
    except Exception as e:
        # 捕获异常，并返回失败结果，包含错误码和异常信息
        return Result.failed(code="E000X", msg=f"document chunk list error {e}")


# 定义一个同步 POST 路由处理函数，用于处理相似查询请求
@router.post("/knowledge/{vector_name}/query")
def similar_query(space_name: str, query_request: KnowledgeQueryRequest):
    print(f"Received params: {space_name}, {query_request}")  # 打印接收到的参数信息
    # 获取 embedding_factory 组件，并创建 VectorStoreConfig 对象
    embedding_factory = CFG.SYSTEM_APP.get_component(
        "embedding_factory", EmbeddingFactory
    )
    config = VectorStoreConfig(
        name=space_name,  # 设置空间名称
        embedding_fn=embedding_factory.create(
            EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]  # 使用指定的嵌入模型配置创建嵌入函数
        ),
    )
    # 创建 VectorStoreConnector 对象，配置类型和配置对象
    vector_store_connector = VectorStoreConnector(
        vector_store_type=CFG.VECTOR_STORE_TYPE,  # 使用指定的向量存储类型
        vector_store_config=config,  # 使用上述创建的 VectorStoreConfig 对象
    )
    # 创建一个嵌入式检索器对象，用于从向量存储中检索与查询相关的顶部K个结果
    retriever = EmbeddingRetriever(
        top_k=query_request.top_k, index_store=vector_store_connector.index_client
    )
    
    # 使用检索器对象检索与查询请求相关的数据块或文档片段
    chunks = retriever.retrieve(query_request.query)
    
    # 创建包含检索到的文本和元数据源的知识查询响应列表
    res = [
        KnowledgeQueryResponse(text=d.content, source=d.metadata["source"])
        for d in chunks
    ]
    
    # 返回一个包含响应列表的字典，键为"response"
    return {"response": res}
@router.post("/knowledge/document/summary")
async def document_summary(request: DocumentSummaryRequest):
    # 打印请求参数信息
    print(f"/document/summary params: {request}")
    try:
        # 使用根追踪器开始一个命名为 "get_chat_instance" 的跨度，类型为 SpanType.CHAT，元数据为请求对象
        with root_tracer.start_span(
            "get_chat_instance", span_type=SpanType.CHAT, metadata=request
        ):
            # 调用知识空间服务的 document_summary 方法，获取聊天信息
            chat = await knowledge_space_service.document_summary(request=request)
        
        # 设置响应头部信息
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Transfer-Encoding": "chunked",
        }
        
        # 导入 StreamingResponse 类
        from starlette.responses import StreamingResponse

        # 如果 chat.prompt_template.stream_out 为 False，则返回不带流的 StreamingResponse
        if not chat.prompt_template.stream_out:
            return StreamingResponse(
                no_stream_generator(chat),
                headers=headers,
                media_type="text/event-stream",
            )
        else:
            # 否则，返回带流的 StreamingResponse，使用 stream_generator 生成器
            return StreamingResponse(
                stream_generator(chat, False, request.model_name),
                headers=headers,
                media_type="text/plain",
            )
    except Exception as e:
        # 捕获异常，并返回失败的 Result 对象，包含错误代码和消息
        return Result.failed(code="E000X", msg=f"document summary error {e}")
```