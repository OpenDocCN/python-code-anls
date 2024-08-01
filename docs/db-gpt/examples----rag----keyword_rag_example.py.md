# `.\DB-GPT-src\examples\rag\keyword_rag_example.py`

```py
import asyncio  # 引入异步IO模块
import os  # 引入操作系统模块

from dbgpt.configs.model_config import ROOT_PATH  # 从dbgpt.configs.model_config模块导入ROOT_PATH常量
from dbgpt.rag import ChunkParameters  # 从dbgpt.rag模块导入ChunkParameters类
from dbgpt.rag.assembler import EmbeddingAssembler  # 从dbgpt.rag.assembler模块导入EmbeddingAssembler类
from dbgpt.rag.knowledge import KnowledgeFactory  # 从dbgpt.rag.knowledge模块导入KnowledgeFactory类
from dbgpt.storage.full_text.elasticsearch import (  # 从dbgpt.storage.full_text.elasticsearch模块导入以下类
    ElasticDocumentConfig,  # ElasticDocumentConfig类
    ElasticDocumentStore,  # ElasticDocumentStore类
)

"""Keyword rag example.
    pre-requirements:
    set your Elasticsearch environment.

    Examples:
        ..code-block:: shell
            python examples/rag/keyword_rag_example.py
"""

def _create_es_connector():
    """Create es connector."""
    # 创建Elasticsearch文档存储的配置对象
    config = ElasticDocumentConfig(
        name="keyword_rag_test",  # 设置文档存储名称
        uri="localhost",  # 设置Elasticsearch URI
        port="9200",  # 设置Elasticsearch端口
        user="elastic",  # 设置Elasticsearch用户名
        password="dbgpt",  # 设置Elasticsearch密码
    )

    return ElasticDocumentStore(config)  # 返回配置对象的文档存储实例


async def main():
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")  # 构建文件路径
    knowledge = KnowledgeFactory.from_file_path(file_path)  # 从文件路径创建知识对象
    keyword_store = _create_es_connector()  # 创建Elasticsearch连接器
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")  # 设置文本分块参数策略
    # 加载知识到嵌入装配器
    assembler = EmbeddingAssembler.load_from_knowledge(
        knowledge=knowledge,  # 使用先前创建的知识对象
        chunk_parameters=chunk_parameters,  # 使用指定的文本分块参数
        index_store=keyword_store,  # 使用Elasticsearch连接器进行索引存储
    )
    assembler.persist()  # 持久化嵌入装配器配置
    # 获取检索器对象
    retriever = assembler.as_retriever(3)  # 设置检索器，参数为返回结果的数量
    chunks = await retriever.aretrieve_with_scores("what is awel talk about", 0.3)  # 使用检索器检索相关文本片段
    print(f"keyword rag example results:{chunks}")  # 打印关键词RAG示例的检索结果


if __name__ == "__main__":
    asyncio.run(main())  # 运行异步主函数
```