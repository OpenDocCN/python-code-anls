# `.\DB-GPT-src\examples\rag\simple_dbschema_retriever_example.py`

```py
"""
AWEL: Simple rag db schema embedding operator example

if you not set vector_store_connector, it will return all tables schema in database.
"""
# 创建一个没有向量存储连接器的数据库模式检索任务
retriever_task = DBSchemaRetrieverOperator(
    connector=_create_temporary_connection()
)

"""
if you set vector_store_connector, it will recall topk similarity tables schema in database.
"""
# 创建一个设置了向量存储连接器的数据库模式检索任务，以检索数据库中相似度排名前k的表模式
retriever_task = DBSchemaRetrieverOperator(
    connector=_create_temporary_connection(),
    top_k=1,
    index_store=vector_store_connector
)

Examples:
    ..code-block:: shell
        curl --location 'http://127.0.0.1:5555/api/v1/awel/trigger/examples/rag/dbschema' \
        --header 'Content-Type: application/json' \
        --data '{"query": "what is user name?"}'
"""

import os
from typing import Dict, List

from dbgpt._private.config import Config
from dbgpt._private.pydantic import BaseModel, Field
from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH
from dbgpt.core import Chunk
from dbgpt.core.awel import DAG, HttpTrigger, JoinOperator, MapOperator
from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector
from dbgpt.rag.embedding import DefaultEmbeddingFactory
from dbgpt.rag.operators import DBSchemaAssemblerOperator, DBSchemaRetrieverOperator
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

CFG = Config()


def _create_vector_connector():
    """Create vector connector."""
    config = ChromaVectorConfig(
        persist_path=os.path.join(PILOT_PATH, "data"),
        name="vector_name",
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),
    )

    return ChromaStore(config)


def _create_temporary_connection():
    """Create a temporary database connection for testing."""
    connect = SQLiteTempConnector.create_temporary_db()
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


def _join_fn(chunks: List[Chunk], query: str) -> str:
    print(f"db schema info is {[chunk.content for chunk in chunks]}")
    return query


class TriggerReqBody(BaseModel):
    query: str = Field(..., description="User query")


class RequestHandleOperator(MapOperator[TriggerReqBody, Dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    async def map(self, input_value: TriggerReqBody) -> Dict:
        # 定义一个异步方法 `map`，接收一个 `TriggerReqBody` 类型的参数 `input_value`，返回一个字典类型
        params = {
            "query": input_value.query,  # 从 input_value 中获取 query 属性，并放入字典 params 中
        }
        # 打印接收到的 input_value 的信息
        print(f"Receive input value: {input_value}")
        # 返回构建好的字典 params
        return params
with DAG("simple_rag_db_schema_example") as dag:
    # 创建一个名为 "simple_rag_db_schema_example" 的 DAG 对象
    trigger = HttpTrigger(
        "/examples/rag/dbschema", methods="POST", request_body=TriggerReqBody
    )
    # 创建一个 HTTP 触发器对象，指定路径为 "/examples/rag/dbschema"，方法为 POST，请求体类型为 TriggerReqBody
    request_handle_task = RequestHandleOperator()
    # 创建一个请求处理操作符对象
    query_operator = MapOperator(lambda request: request["query"])
    # 创建一个映射操作符对象，对输入请求中的 "query" 字段执行映射操作
    index_store = _create_vector_connector()
    # 调用 _create_vector_connector() 函数创建一个向量连接器对象
    connector = _create_temporary_connection()
    # 调用 _create_temporary_connection() 函数创建一个临时连接对象
    assembler_task = DBSchemaAssemblerOperator(
        connector=connector,
        index_store=index_store,
    )
    # 创建一个数据库模式装配器操作符对象，指定连接器和索引存储对象作为参数
    join_operator = JoinOperator(combine_function=_join_fn)
    # 创建一个连接操作符对象，指定连接函数 _join_fn 作为参数
    retriever_task = DBSchemaRetrieverOperator(
        connector=_create_temporary_connection(),
        top_k=1,
        index_store=index_store,
    )
    # 创建一个数据库模式检索操作符对象，指定临时连接对象、top_k 值为 1 和索引存储对象作为参数
    result_parse_task = MapOperator(lambda chunks: [chunk.content for chunk in chunks])
    # 创建一个结果解析操作符对象，对输入的数据块列表执行解析操作，提取内容字段
    trigger >> assembler_task >> join_operator
    # 设置 DAG 的工作流程：触发器触发后执行装配器操作，然后执行连接操作
    trigger >> request_handle_task >> query_operator >> join_operator
    # 设置 DAG 的工作流程：触发器触发后执行请求处理操作，然后执行映射操作，最后执行连接操作
    join_operator >> retriever_task >> result_parse_task
    # 设置 DAG 的工作流程：连接操作完成后执行检索操作，最后执行结果解析操作

if __name__ == "__main__":
    if dag.leaf_nodes[0].dev_mode:
        # 如果 DAG 的第一个叶节点处于开发模式
        # 开发模式下，可以在本地运行 DAG 进行调试。
        from dbgpt.core.awel import setup_dev_environment
        setup_dev_environment([dag], port=5555)
    else:
        pass
```