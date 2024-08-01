# `.\DB-GPT-src\tests\unit_tests\embedding_engine\url_test.py`

```py
# 导入必要的模块和类
from dbgpt import EmbeddingEngine, KnowledgeType

# 定义一个 URL 变量，指向 DBGPT 文档的地址
url = "https://docs.dbgpt.site/docs/overview"

# 设定嵌入模型的名称
embedding_model = "your_embedding_model"

# 定义向量存储的类型为 Chroma
vector_store_type = "Chroma"

# 指定 Chroma 向量存储的持久化路径
chroma_persist_path = "your_persist_path"

# 创建向量存储的配置字典，包括存储名称、存储类型和持久化路径
vector_store_config = {
    "vector_store_name": url.replace(":", ""),  # 将 URL 中的冒号替换为空，作为存储名称
    "vector_store_type": vector_store_type,     # 使用之前定义的向量存储类型
    "chroma_persist_path": chroma_persist_path  # 使用之前定义的持久化路径
}

# 创建嵌入引擎对象，传入知识源的 URL、知识类型、模型名称和向量存储配置
embedding_engine = EmbeddingEngine(
    knowledge_source=url,
    knowledge_type=KnowledgeType.URL.value,
    model_name=embedding_model,
    vector_store_config=vector_store_config,
)

# 对知识源的内容进行嵌入，存储到向量存储中
embedding_engine.knowledge_embedding()
```