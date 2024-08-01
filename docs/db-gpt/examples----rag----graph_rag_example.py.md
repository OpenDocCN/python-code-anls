# `.\DB-GPT-src\examples\rag\graph_rag_example.py`

```py
import asyncio  # 导入异步 I/O 库 asyncio
import os  # 导入操作系统相关功能模块

from dbgpt.configs.model_config import ROOT_PATH  # 从 dbgpt.configs.model_config 模块导入 ROOT_PATH 变量
from dbgpt.model.proxy.llms.chatgpt import OpenAILLMClient  # 从 dbgpt.model.proxy.llms.chatgpt 模块导入 OpenAILLMClient 类
from dbgpt.rag import ChunkParameters  # 从 dbgpt.rag 模块导入 ChunkParameters 类
from dbgpt.rag.assembler import EmbeddingAssembler  # 从 dbgpt.rag.assembler 模块导入 EmbeddingAssembler 类
from dbgpt.rag.knowledge import KnowledgeFactory  # 从 dbgpt.rag.knowledge 模块导入 KnowledgeFactory 类
from dbgpt.rag.retriever import RetrieverStrategy  # 从 dbgpt.rag.retriever 模块导入 RetrieverStrategy 类
from dbgpt.storage.knowledge_graph.knowledge_graph import (  # 从 dbgpt.storage.knowledge_graph.knowledge_graph 模块导入以下类
    BuiltinKnowledgeGraph,
    BuiltinKnowledgeGraphConfig,
)

"""GraphRAG example.
    pre-requirements:
    * Set LLM config (url/sk) in `.env`.
    * Setup/startup TuGraph from: https://github.com/TuGraph-family/tugraph-db
    * Config TuGraph following the format below. 
    ```
    GRAPH_STORE_TYPE=TuGraph
    TUGRAPH_HOST=127.0.0.1
    TUGRAPH_PORT=7687
    TUGRAPH_USERNAME=admin
    TUGRAPH_PASSWORD=73@TuGraph
    ```py

    Examples:
        ..code-block:: shell
            python examples/rag/graph_rag_example.py
"""

def _create_kg_connector():
    """Create knowledge graph connector."""
    # 创建一个内置知识图谱对象，使用指定的配置参数
    return BuiltinKnowledgeGraph(
        config=BuiltinKnowledgeGraphConfig(
            name="graph_rag_test",
            embedding_fn=None,
            llm_client=OpenAILLMClient(),
            model_name="gpt-3.5-turbo",
        ),
    )

async def main():
    # 拼接文件路径，获取要处理的文件路径
    file_path = os.path.join(ROOT_PATH, "examples/test_files/tranformers_story.md")
    # 从文件路径中读取知识数据
    knowledge = KnowledgeFactory.from_file_path(file_path)
    # 创建知识图谱连接器
    graph_store = _create_kg_connector()
    # 定义块参数对象，使用 CHUNK_BY_SIZE 策略
    chunk_parameters = ChunkParameters(chunk_strategy="CHUNK_BY_SIZE")
    # 获取嵌入组装器对象，从知识数据中加载
    assembler = await EmbeddingAssembler.aload_from_knowledge(
        knowledge=knowledge,
        chunk_parameters=chunk_parameters,
        index_store=graph_store,
        retrieve_strategy=RetrieverStrategy.GRAPH,
    )
    # 持久化组装器状态
    await assembler.apersist()
    # 获取嵌入检索器对象
    retriever = assembler.as_retriever(3)
    # 使用检索器对象检索带有得分的块
    chunks = await retriever.aretrieve_with_scores(
        "What actions has Megatron taken ?", score_threshold=0.3
    )
    # 打印检索结果
    print(f"embedding rag example results:{chunks}")
    # 删除指定名称的向量数据
    graph_store.delete_vector_name("graph_rag_test")

if __name__ == "__main__":
    # 运行主程序入口
    asyncio.run(main())
```