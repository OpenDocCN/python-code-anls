# `.\DB-GPT-src\examples\rag\cross_encoder_rerank_example.py`

```py
"""This example demonstrates how to use the cross-encoder reranker
to rerank the retrieved chunks.
The cross-encoder reranker is a neural network model that takes a query
and a chunk as input and outputs a score that represents the relevance of the chunk
to the query.

Download pretrained cross-encoder models can be found at https://huggingface.co/models.
Example:
    python examples/rag/cross_encoder_rerank_example.py
"""
import asyncio
import os

# 导入所需的模块和类
from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH, ROOT_PATH
from dbgpt.rag import ChunkParameters
from dbgpt.rag.assembler import EmbeddingAssembler
from dbgpt.rag.embedding import DefaultEmbeddingFactory
from dbgpt.rag.knowledge import KnowledgeFactory
from dbgpt.rag.retriever.rerank import CrossEncoderRanker
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig


def _create_vector_connector():
    """Create vector connector."""
    # 创建用于连接向量的配置对象
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,  # 持久化路径
        name="embedding_rag_test",  # 名称
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),  # 创建默认嵌入模型
    )

    return ChromaStore(config)  # 返回向量存储对象


async def main():
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")
    knowledge = KnowledgeFactory.from_file_path(file_path)  # 从文件路径创建知识对象
    vector_connector = _create_vector_connector()  # 创建向量连接器
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_MARKDOWN_HEADER")  # 设置分块参数
    # 获取嵌入装配器
    assembler = EmbeddingAssembler.load_from_knowledge(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        index_store=vector_connector,
    )
    assembler.persist()  # 持久化装配器
    # 获取嵌入检索器
    retriever = assembler.as_retriever(3)  # 转换为检索器，指定 topk 值为 3
    # 创建元数据过滤器
    query = "what is awel talk about"  # 查询语句
    chunks = await retriever.aretrieve_with_scores(query, 0.3)  # 检索带有分数的分块

    print("before rerank results:\n")
    for i, chunk in enumerate(chunks):
        print(f"----{i+1}.chunk content:{chunk.content}\n score:{chunk.score}")
    # 使用交叉编码器重新排序
    cross_encoder_model = os.path.join(MODEL_PATH, "bge-reranker-base")  # 加载交叉编码器模型
    rerank = CrossEncoderRanker(topk=3, model=cross_encoder_model)  # 创建交叉编码器重新排序对象
    new_chunks = rerank.rank(chunks, query=query)  # 对分块进行重新排序
    print("after cross-encoder rerank results:\n")
    for i, chunk in enumerate(new_chunks):
        print(f"----{i+1}.chunk content:{chunk.content}\n score:{chunk.score}")


if __name__ == "__main__":
    asyncio.run(main())  # 运行主函数
```