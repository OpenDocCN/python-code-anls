# `.\DB-GPT-src\examples\rag\bm25_retriever_example.py`

```py
# 引入 asyncio 库，用于异步编程
import asyncio
# 引入 os 库，提供与操作系统交互的功能
import os

# 从 dbgpt.configs.model_config 中导入 ROOT_PATH 变量
from dbgpt.configs.model_config import ROOT_PATH
# 从 dbgpt.rag 中导入 ChunkParameters 类
from dbgpt.rag import ChunkParameters
# 从 dbgpt.rag.assembler.bm25 中导入 BM25Assembler 类
from dbgpt.rag.assembler.bm25 import BM25Assembler
# 从 dbgpt.rag.knowledge 中导入 KnowledgeFactory 类
from dbgpt.rag.knowledge import KnowledgeFactory
# 从 dbgpt.storage.vector_store.elastic_store 中导入 ElasticsearchVectorConfig 类
from dbgpt.storage.vector_store.elastic_store import ElasticsearchVectorConfig

"""Embedding rag example.
    pre-requirements:
    set your elasticsearch config in your example code.

    Examples:
        ..code-block:: shell
            python examples/rag/bm25_retriever_example.py
"""
# 此处是多行字符串，用于说明代码示例的用途和预备条件

def _create_es_config():
    """Create vector connector."""
    # 创建并返回 ElasticsearchVectorConfig 实例，用于连接到 Elasticsearch
    return ElasticsearchVectorConfig(
        name="bm25_es_dbgpt",
        uri="localhost",
        port="9200",
        user="elastic",
        password="dbgpt",
    )


async def main():
    # 构建文件路径，使用 ROOT_PATH 变量和文件相对路径
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")
    # 从文件路径中加载知识库，返回 KnowledgeFactory 实例
    knowledge = KnowledgeFactory.from_file_path(file_path)
    # 创建 Elasticsearch 连接配置
    es_config = _create_es_config()
    # 创建块参数，使用 CHUNK_BY_SIZE 策略
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")
    # 创建 BM25Assembler 实例，从知识库加载数据，并使用给定的 Elasticsearch 配置和块参数
    assembler = BM25Assembler.load_from_knowledge(
        knowledge=knowledge,
        es_config=es_config,
        chunk_parameters=chunk_parameters,
    )
    # 持久化 BM25Assembler 实例
    assembler.persist()
    # 获取 BM25 检索器
    retriever = assembler.as_retriever(3)
    # 使用检索器检索指定查询的结果和分数
    chunks = retriever.retrieve_with_scores("what is awel talk about", 0.3)
    # 打印 BM25 检索示例的结果
    print(f"bm25 rag example results:{chunks}")


if __name__ == "__main__":
    # 运行主函数 main()，使用 asyncio 库进行异步执行
    asyncio.run(main())
```