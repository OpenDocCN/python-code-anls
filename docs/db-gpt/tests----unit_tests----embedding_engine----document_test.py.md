# `.\DB-GPT-src\tests\unit_tests\embedding_engine\document_test.py`

```py
from dbgpt import EmbeddingEngine, KnowledgeType  # 导入必要的模块和类

embedding_model = "your_embedding_model"  # 设置嵌入模型的名称
vector_store_type = "Chroma"  # 设置向量存储类型为 Chroma
chroma_persist_path = "your_persist_path"  # 设置 Chroma 持久化路径
vector_store_config = {
    "vector_store_name": "document_test",  # 设置向量存储的名称
    "vector_store_type": vector_store_type,  # 使用上面定义的向量存储类型
    "chroma_persist_path": chroma_persist_path,  # 使用上面定义的持久化路径
}

# it can be .md,.pdf,.docx, .csv, .html
document_path = "your_path/test.md"  # 设置待处理文档的路径
embedding_engine = EmbeddingEngine(
    knowledge_source=document_path,  # 指定知识源为上述文档路径
    knowledge_type=KnowledgeType.DOCUMENT.value,  # 设置知识类型为文档类型
    model_name=embedding_model,  # 使用上面定义的嵌入模型名称
    vector_store_config=vector_store_config,  # 使用上面定义的向量存储配置
)
# embedding document content to vector store
embedding_engine.knowledge_embedding()  # 执行文档内容嵌入到向量存储中的操作
```