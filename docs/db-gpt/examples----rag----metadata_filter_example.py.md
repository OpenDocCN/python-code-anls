# `.\DB-GPT-src\examples\rag\metadata_filter_example.py`

```py
"""Rag Metadata Properties filter example.
pre-requirements:
make sure you have set your embedding model path in your example code.

Examples:
..code-block:: shell
python examples/rag/metadata_filter_example.py
"""
import asyncio  # 导入 asyncio 库，用于异步编程
import os  # 导入 os 库，用于处理操作系统相关的功能

from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH, ROOT_PATH  # 从 dbgpt.configs.model_config 导入相关路径常量
from dbgpt.rag import ChunkParameters  # 从 dbgpt.rag 导入 ChunkParameters 类
from dbgpt.rag.assembler import EmbeddingAssembler  # 从 dbgpt.rag.assembler 导入 EmbeddingAssembler 类
from dbgpt.rag.embedding import DefaultEmbeddingFactory  # 从 dbgpt.rag.embedding 导入 DefaultEmbeddingFactory 类
from dbgpt.rag.knowledge import KnowledgeFactory  # 从 dbgpt.rag.knowledge 导入 KnowledgeFactory 类
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig  # 从 dbgpt.storage.vector_store.chroma_store 导入 ChromaStore 和 ChromaVectorConfig 类
from dbgpt.storage.vector_store.filters import MetadataFilter, MetadataFilters  # 从 dbgpt.storage.vector_store.filters 导入 MetadataFilter 和 MetadataFilters 类


def _create_vector_connector():
    """Create vector connector."""
    # 创建 ChromaVectorConfig 对象，配置持久化路径、名称以及默认模型路径
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,
        name="metadata_rag_test",
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),
    )

    return ChromaStore(config)  # 返回基于 config 配置的 ChromaStore 对象


async def main():
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")  # 构建文件路径
    knowledge = KnowledgeFactory.from_file_path(file_path)  # 从文件路径创建知识库对象
    vector_store = _create_vector_connector()  # 创建连接到矢量存储的对象
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_MARKDOWN_HEADER")  # 设置分块参数，以 Markdown 标题分块
    # 从知识库加载 EmbeddingAssembler 对象
    assembler = EmbeddingAssembler.load_from_knowledge(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        index_store=vector_store,
    )
    assembler.persist()  # 持久化 EmbeddingAssembler 对象
    retriever = assembler.as_retriever(3)  # 从 assembler 中获取检索器，设置最大检索数为 3
    # 创建 MetadataFilter 对象，指定键为 "Header2"，值为 "AWEL Design"
    metadata_filter = MetadataFilter(key="Header2", value="AWEL Design")
    filters = MetadataFilters(filters=[metadata_filter])  # 创建 MetadataFilters 对象，包含一个 MetadataFilter
    # 使用检索器和过滤器进行带有分数的检索
    chunks = await retriever.aretrieve_with_scores(
        "what is awel talk about", 0.0, filters
    )
    print(f"embedding rag example results:{chunks}")  # 打印检索结果
    vector_store.delete_vector_name("metadata_rag_test")  # 删除指定名称的矢量


if __name__ == "__main__":
    asyncio.run(main())  # 运行主函数 main()，使用 asyncio 执行异步任务
```