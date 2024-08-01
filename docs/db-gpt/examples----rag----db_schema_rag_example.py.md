# `.\DB-GPT-src\examples\rag\db_schema_rag_example.py`

```py
# 导入必要的模块
import os

# 导入模型路径和 PILOT_PATH
from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH
# 导入 SQLiteTempConnector 类
from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector
# 导入 DBSchemaAssembler 类
from dbgpt.rag.assembler import DBSchemaAssembler
# 导入 DefaultEmbeddingFactory 类
from dbgpt.rag.embedding import DefaultEmbeddingFactory
# 导入 ChromaStore 和 ChromaVectorConfig 类
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

"""DB struct rag example.
    pre-requirements:
    set your embedding model path in your example code.
    ```
    embedding_model_path = "{your_embedding_model_path}"
    ```py
    
    Examples:
        ..code-block:: shell
            python examples/rag/db_schema_rag_example.py
"""

# 创建临时数据库连接用于测试
def _create_temporary_connection():
    """Create a temporary database connection for testing."""
    # 创建临时数据库连接
    connect = SQLiteTempConnector.create_temporary_db()
    # 创建临时数据库表
    connect.create_temp_tables(
        {
            "user": {
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "name": "TEXT",
                    "age": "INTEGER",
                },
                "data": [
                    (1, "Tom", 10),
                    (2, "Jerry", 16),
                    (3, "Jack", 18),
                    (4, "Alice", 20),
                    (5, "Bob", 22),
                ],
            }
        }
    )
    return connect

# 创建向量连接器
def _create_vector_connector():
    """Create vector connector."""
    # 配置向量连接器
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,
        name="dbschema_rag_test",
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),
    )

    return ChromaStore(config)

# 主函数
if __name__ == "__main__":
    # 创建临时数据库连接
    connection = _create_temporary_connection()
    # 创建向量连接器
    index_store = _create_vector_connector()
    # 从连接中加载 DBSchemaAssembler
    assembler = DBSchemaAssembler.load_from_connection(
        connector=connection,
        index_store=index_store,
    )
    # 持久化 DBSchemaAssembler
    assembler.persist()
    # 获取数据库模式检索器
    retriever = assembler.as_retriever(top_k=1)
    # 检索数据库模式
    chunks = retriever.retrieve("show columns from user")
    # 打印结果
    print(f"db schema rag example results:{[chunk.content for chunk in chunks]}")
    # 删除向量名称
    index_store.delete_vector_name("dbschema_rag_test")
```