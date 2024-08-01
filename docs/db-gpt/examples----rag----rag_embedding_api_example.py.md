# `.\DB-GPT-src\examples\rag\rag_embedding_api_example.py`

```py
"""A RAG example using the OpenAPIEmbeddings.

Example:

    Test with `OpenAI embeddings
    <https://platform.openai.com/docs/api-reference/embeddings/create>`_.

    .. code-block:: shell

        export API_SERVER_BASE_URL=${OPENAI_API_BASE:-"https://api.openai.com/v1"}
        export API_SERVER_API_KEY="${OPENAI_API_KEY}"
        export API_SERVER_EMBEDDINGS_MODEL="text-embedding-ada-002"
        python examples/rag/rag_embedding_api_example.py

    Test with DB-GPT `API Server
    <https://docs.dbgpt.site/docs/installation/advanced_usage/OpenAI_SDK_call#start-apiserver>`_.

    .. code-block:: shell
        export API_SERVER_BASE_URL="http://localhost:8100/api/v1"
        export API_SERVER_API_KEY="your_api_key"
        export API_SERVER_EMBEDDINGS_MODEL="text2vec"
        python examples/rag/rag_embedding_api_example.py

"""
import asyncio  # 导入异步 I/O 库 asyncio
import os  # 导入操作系统相关功能
from typing import Optional  # 引入类型提示

from dbgpt.configs.model_config import PILOT_PATH, ROOT_PATH  # 从 dbgpt 库导入配置
from dbgpt.rag import ChunkParameters  # 导入 ChunkParameters 类
from dbgpt.rag.assembler import EmbeddingAssembler  # 导入 EmbeddingAssembler 类
from dbgpt.rag.embedding import OpenAPIEmbeddings  # 导入 OpenAPIEmbeddings 类
from dbgpt.rag.knowledge import KnowledgeFactory  # 导入 KnowledgeFactory 类
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig  # 导入 ChromaStore 和 ChromaVectorConfig

def _create_embeddings(
    api_url: str = None, api_key: Optional[str] = None, model_name: Optional[str] = None
) -> OpenAPIEmbeddings:
    """创建 OpenAPIEmbeddings 对象。

    根据提供的 API URL、API Key 和模型名称创建 OpenAPIEmbeddings 对象。

    Args:
        api_url (str, optional): API 的基础 URL 地址，默认从环境变量获取。
        api_key (Optional[str], optional): API 的访问密钥，默认从环境变量获取。
        model_name (Optional[str], optional): 使用的嵌入模型名称，默认从环境变量获取。

    Returns:
        OpenAPIEmbeddings: 返回配置好的 OpenAPIEmbeddings 对象。
    """
    if not api_url:
        api_server_base_url = os.getenv(
            "API_SERVER_BASE_URL", "http://localhost:8100/api/v1/"
        )
        api_url = f"{api_server_base_url}/embeddings"
    if not api_key:
        api_key = os.getenv("API_SERVER_API_KEY")

    if not model_name:
        model_name = os.getenv("API_SERVER_EMBEDDINGS_MODEL", "text2vec")

    return OpenAPIEmbeddings(api_url=api_url, api_key=api_key, model_name=model_name)


def _create_vector_connector():
    """创建向量存储连接器。

    使用配置文件创建 ChromaStore 对象，用于存储和管理向量。

    Returns:
        ChromaStore: 返回配置好的 ChromaStore 对象。
    """
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,
        name="embedding_api_rag_test",
        embedding_fn=_create_embeddings(),
    )

    return ChromaStore(config)


async def main():
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")
    knowledge = KnowledgeFactory.from_file_path(file_path)  # 从文件路径创建知识库对象
    vector_store = _create_vector_connector()  # 创建向量存储连接器
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")  # 设置分块参数
    assembler = EmbeddingAssembler.load_from_knowledge(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        index_store=vector_store,
    )  # 从知识库加载 EmbeddingAssembler 对象
    assembler.persist()  # 持久化处理
    retriever = assembler.as_retriever(3)  # 将 assembler 转换为检索器对象
    chunks = await retriever.aretrieve_with_scores("what is awel talk about", 0.3)  # 使用检索器获取相关内容
    print(f"embedding rag example results:{chunks}")  # 打印结果
    vector_store.delete_vector_name("embedding_api_rag_test")  # 删除向量存储中的名称

if __name__ == "__main__":
    asyncio.run(main())  # 运行主函数，使用 asyncio 进行异步处理
```