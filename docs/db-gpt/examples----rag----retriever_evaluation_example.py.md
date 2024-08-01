# `.\DB-GPT-src\examples\rag\retriever_evaluation_example.py`

```py
import asyncio
import os
from typing import Optional

from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH, ROOT_PATH
from dbgpt.core import Embeddings
from dbgpt.rag import ChunkParameters
from dbgpt.rag.assembler import EmbeddingAssembler
from dbgpt.rag.embedding import DefaultEmbeddingFactory
from dbgpt.rag.evaluation import RetrieverEvaluator
from dbgpt.rag.evaluation.retriever import (
    RetrieverHitRateMetric,
    RetrieverMRRMetric,
    RetrieverSimilarityMetric,
)
from dbgpt.rag.knowledge import KnowledgeFactory
from dbgpt.rag.operators import EmbeddingRetrieverOperator
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig


def _create_embeddings(
    model_name: Optional[str] = "text2vec-large-chinese",
) -> Embeddings:
    """Create embeddings."""
    # 返回一个默认模型名为指定模型名的嵌入对象
    return DefaultEmbeddingFactory(
        default_model_name=os.path.join(MODEL_PATH, model_name),
    ).create()


def _create_vector_connector(embeddings: Embeddings):
    """Create vector connector."""
    # 创建一个 ChromaVectorConfig 对象，配置文件存储路径为 PILOT_PATH，名称为 "embedding_rag_test"，使用给定的嵌入函数
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,
        name="embedding_rag_test",
        embedding_fn=embeddings,
    )

    return ChromaStore(config)


async def main():
    # 定义文件路径为 ROOT_PATH 下的指定文件
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")
    # 从指定文件路径加载知识库对象
    knowledge = KnowledgeFactory.from_file_path(file_path)
    # 创建嵌入对象
    embeddings = _create_embeddings()
    # 创建向量存储连接器对象
    vector_connector = _create_vector_connector(embeddings)
    # 设置块参数为根据 Markdown 标题进行分块的策略
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_MARKDOWN_HEADER")
    
    # 从知识库加载嵌入组装器对象，使用给定的知识、块参数和索引存储
    assembler = EmbeddingAssembler.load_from_knowledge(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        index_store=vector_connector,
    )
    # 持久化嵌入组装器对象
    assembler.persist()

    # 定义数据集，包含一个查询和相关上下文
    dataset = [
        {
            "query": "what is awel talk about",
            "contexts": [
                "# What is AWEL? \n\nAgentic Workflow Expression Language(AWEL) is a "
                "set of intelligent agent workflow expression language specially "
                "designed for large model application\ndevelopment. It provides great "
                "functionality and flexibility. Through the AWEL API, you can focus on "
                "the development of business logic for LLMs applications\nwithout "
                "paying attention to cumbersome model and environment details.\n\nAWEL "
                "adopts a layered API design. AWEL's layered API design architecture is "
                "shown in the figure below."
            ],
        },
    ]
    # 创建检索器评估器对象，使用指定的操作器类、嵌入对象和操作器参数
    evaluator = RetrieverEvaluator(
        operator_cls=EmbeddingRetrieverOperator,
        embeddings=embeddings,
        operator_kwargs={
            "top_k": 5,
            "index_store": vector_connector,
        },
    )
    # 定义评估指标列表
    metrics = [
        RetrieverHitRateMetric(),
        RetrieverMRRMetric(),
        RetrieverSimilarityMetric(embeddings=embeddings),
    ]
    # 执行数据集的评估，并返回结果
    results = await evaluator.evaluate(dataset, metrics)
    # 遍历 results 列表中的每个 result 对象
    for result in results:
        # 遍历每个 result 对象中的 metric 对象
        for metric in result:
            # 打印 metric 对象的指标名称
            print("Metric:", metric.metric_name)
            # 打印 metric 对象的查询问题
            print("Question:", metric.query)
            # 打印 metric 对象的分数
            print("Score:", metric.score)
    # 打印整体结果 results，格式化输出在新行上
    print(f"Results:\n{results}")
# 如果这个脚本是直接被运行的主程序入口
if __name__ == "__main__":
    # 使用 asyncio 模块运行 main() 函数
    asyncio.run(main())
```