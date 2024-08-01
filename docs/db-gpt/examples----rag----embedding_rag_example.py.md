# `.\DB-GPT-src\examples\rag\embedding_rag_example.py`

```py
# 引入 asyncio 和 os 模块
import asyncio
import os

# 从指定路径导入模型配置和根路径
from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH, ROOT_PATH
# 从 dbgpt.rag 中导入 ChunkParameters 类
from dbgpt.rag import ChunkParameters
# 从 dbgpt.rag.assembler 中导入 EmbeddingAssembler 类
from dbgpt.rag.assembler import EmbeddingAssembler
# 从 dbgpt.rag.embedding 中导入 DefaultEmbeddingFactory 类
from dbgpt.rag.embedding import DefaultEmbeddingFactory
# 从 dbgpt.rag.knowledge 中导入 KnowledgeFactory 类
from dbgpt.rag.knowledge import KnowledgeFactory
# 从 dbgpt.storage.vector_store.chroma_store 中导入 ChromaStore 和 ChromaVectorConfig 类
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

# 多行字符串，用于说明示例代码的预先要求和用法示例
"""
Embedding rag example.
    pre-requirements:
    set your embedding model path in your example code.
    ```
    embedding_model_path = "{your_embedding_model_path}"
    ```py

    Examples:
        ..code-block:: shell
            python examples/rag/embedding_rag_example.py
"""

# 创建向量连接器的函数
def _create_vector_connector():
    # 配置 ChromaVectorConfig 对象
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,  # 持久化路径设为 PILOT_PATH
        name="embedding_rag_test",  # 名称设为 "embedding_rag_test"
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),  # 创建默认嵌入模型名称为 text2vec-large-chinese 的工厂对象并设为 embedding_fn
    )

    return ChromaStore(config)  # 返回基于 config 配置的 ChromaStore 对象


# 异步主函数
async def main():
    # 组合文件路径为 ROOT_PATH 下的 "docs/docs/awel/awel.md"
    file_path = os.path.join(ROOT_PATH, "docs/docs/awel/awel.md")
    # 从文件路径加载知识数据到 KnowledgeFactory 对象
    knowledge = KnowledgeFactory.from_file_path(file_path)
    # 创建向量存储连接器
    vector_store = _create_vector_connector()
    # 配置块参数对象为 CHUNK_BY_SIZE 策略
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")
    # 从知识数据和向量存储加载 EmbeddingAssembler 对象
    assembler = EmbeddingAssembler.load_from_knowledge(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        index_store=vector_store,
    )
    assembler.persist()  # 持久化装配器状态
    # 获取装配器的嵌入检索器，参数为检索数量 3
    retriever = assembler.as_retriever(3)
    # 异步检索包含查询文本 "what is awel talk about" 的块，并返回结果
    chunks = await retriever.aretrieve_with_scores("what is awel talk about", 0.3)
    print(f"embedding rag example results:{chunks}")  # 打印嵌入 rag 示例结果


if __name__ == "__main__":
    asyncio.run(main())  # 运行异步主函数 main
```