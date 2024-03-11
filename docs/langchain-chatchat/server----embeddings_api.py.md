# `.\Langchain-Chatchat\server\embeddings_api.py`

```
# 导入所需的模块和类
from langchain.docstore.document import Document
from configs import EMBEDDING_MODEL, logger
from server.model_workers.base import ApiEmbeddingsParams
from server.utils import BaseResponse, get_model_worker_config, list_embed_models, list_online_embed_models
from fastapi import Body
from fastapi.concurrency import run_in_threadpool
from typing import Dict, List

# 获取在线可用的Embeddings模型列表
online_embed_models = list_online_embed_models()

# 定义函数用于对文本进行向量化处理
def embed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> BaseResponse:
    '''
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    '''
    try:
        # 检查是否使用本地Embeddings模型
        if embed_model in list_embed_models():
            # 导入加载本地Embeddings模型的函数
            from server.utils import load_local_embeddings
            # 加载本地Embeddings模型
            embeddings = load_local_embeddings(model=embed_model)
            # 返回文本向量化结果
            return BaseResponse(data=embeddings.embed_documents(texts))

        # 检查是否使用在线API
        if embed_model in list_online_embed_models():
            # 获取指定Embeddings模型的配置信息
            config = get_model_worker_config(embed_model)
            worker_class = config.get("worker_class")
            embed_model = config.get("embed_model")
            worker = worker_class()
            # 检查工作类是否支持向量化
            if worker_class.can_embedding():
                # 设置API请求参数
                params = ApiEmbeddingsParams(texts=texts, to_query=to_query, embed_model=embed_model)
                # 执行向量化操作
                resp = worker.do_embeddings(params)
                # 返回向量化结果
                return BaseResponse(**resp)

        # 返回错误信息，指定的模型不支持Embeddings功能
        return BaseResponse(code=500, msg=f"指定的模型 {embed_model} 不支持 Embeddings 功能。")
    except Exception as e:
        # 记录错误日志
        logger.error(e)
        # 返回错误信息
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")

# 定义异步函数用于对文本进行向量化处理
async def aembed_texts(
    texts: List[str],
    embed_model: str = EMBEDDING_MODEL,
    to_query: bool = False,
) -> BaseResponse:
    '''
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    '''
    # 尝试执行以下代码块，捕获可能出现的异常
    try:
        # 如果指定的嵌入模型在本地嵌入模型列表中
        if embed_model in list_embed_models(): 
            # 从server.utils模块中导入load_local_embeddings函数
            from server.utils import load_local_embeddings
            # 载入本地嵌入模型
            embeddings = load_local_embeddings(model=embed_model)
            # 返回嵌入文档的基本响应数据
            return BaseResponse(data=await embeddings.aembed_documents(texts))

        # 如果指定的嵌入模型在在线嵌入模型列表中
        if embed_model in list_online_embed_models(): 
            # 在线API中运行embed_texts函数
            return await run_in_threadpool(embed_texts,
                                           texts=texts,
                                           embed_model=embed_model,
                                           to_query=to_query)
    # 捕获任何异常并将其存储在变量e中
    except Exception as e:
        # 记录错误日志
        logger.error(e)
        # 返回包含错误信息的基本响应对象
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")
# 定义一个接受文本列表并返回向量化结果的端点函数
def embed_texts_endpoint(
        texts: List[str] = Body(..., description="要嵌入的文本列表", examples=[["hello", "world"]]),
        embed_model: str = Body(EMBEDDING_MODEL,
                                description=f"使用的嵌入模型，除了本地部署的Embedding模型，也支持在线API({online_embed_models})提供的嵌入服务。"),
        to_query: bool = Body(False, description="向量是否用于查询。有些模型如Minimax对存储/查询的向量进行了区分优化。"),
) -> BaseResponse:
    '''
    对文本进行向量化，返回 BaseResponse(data=List[List[float]])
    '''
    # 调用 embed_texts 函数对文本进行向量化，并返回结果
    return embed_texts(texts=texts, embed_model=embed_model, to_query=to_query)


# 将文档列表向量化的函数
def embed_documents(
        docs: List[Document],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> Dict:
    """
    将 List[Document] 向量化，转化为 VectorStore.add_embeddings 可以接受的参数
    """
    # 从文档列表中提取文本内容和元数据
    texts = [x.page_content for x in docs]
    metadatas = [x.metadata for x in docs]
    # 调用 embed_texts 函数对文本进行向量化，并获取向量化结果
    embeddings = embed_texts(texts=texts, embed_model=embed_model, to_query=to_query).data
    # 如果向量化结果不为空，则返回包含文本、向量和元数据的字典
    if embeddings is not None:
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
```