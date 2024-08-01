# `.\DB-GPT-src\examples\awel\simple_nl_schema_sql_chart_example.py`

```py
# 导入必要的模块
import os  # 导入操作系统模块
from typing import Any, Dict, Optional  # 导入类型提示相关模块

from pandas import DataFrame  # 从 pandas 库中导入 DataFrame 类

# 导入 Pydantic 相关模块
from dbgpt._private.pydantic import BaseModel, Field
# 导入模型配置路径
from dbgpt.configs.model_config import MODEL_PATH, PILOT_PATH
# 导入 dbgpt 核心模块
from dbgpt.core import LLMClient, ModelMessage, ModelMessageRoleType, ModelRequest
# 导入 AWEL 相关模块
from dbgpt.core.awel import DAG, HttpTrigger, JoinOperator, MapOperator
# 导入 RDBMS 连接器基础类
from dbgpt.datasource.rdbms.base import RDBMSConnector
# 导入 SQLite 临时连接器
from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector
# 导入 OpenAI 的语言模型客户端代理
from dbgpt.model.proxy import OpenAILLMClient
# 导入默认嵌入工厂
from dbgpt.rag.embedding import DefaultEmbeddingFactory
# 导入模式链接操作符
from dbgpt.rag.operators.schema_linking import SchemaLinkingOperator
# 导入矢量存储类及配置
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig
# 导入异步任务运行工具
from dbgpt.util.chat_util import run_async_tasks

"""AWEL: Simple nl-schemalinking-sql-chart operator example

    pre-requirements:
        1. install openai python sdk
        ```
            pip install "db-gpt[openai]"
        ```py
        2. set openai key and base
        ```
            export OPENAI_API_KEY={your_openai_key}
            export OPENAI_API_BASE={your_openai_base}
        ```py
        or
        ```
            import os
            os.environ["OPENAI_API_KEY"] = {your_openai_key}
            os.environ["OPENAI_API_BASE"] = {your_openai_base}
        ```py
        python examples/awel/simple_nl_schema_sql_chart_example.py
    Examples:
        ..code-block:: shell
        curl --location 'http://127.0.0.1:5555/api/v1/awel/trigger/examples/rag/schema_linking' \
--header 'Content-Type: application/json' \
--data '{"query": "Statistics of user age in the user table are based on three categories: age is less than 10, age is greater than or equal to 10 and less than or equal to 20, and age is greater than 20. The first column of the statistical results is different ages, and the second column is count."}' 
"""

INSTRUCTION = (
    "I want you to act as a SQL terminal in front of an example database, you need only to return the sql "
    "command to me.Below is an instruction that describes a task, Write a response that appropriately "
    "completes the request.\n###Instruction:\n{}"
)
INPUT_PROMPT = "\n###Input:\n{}\n###Response:"


def _create_vector_connector():
    """Create vector connector."""
    # 创建 ChromaVectorConfig 对象，用于配置矢量存储
    config = ChromaVectorConfig(
        persist_path=PILOT_PATH,
        name="embedding_rag_test",
        embedding_fn=DefaultEmbeddingFactory(
            default_model_name=os.path.join(MODEL_PATH, "text2vec-large-chinese"),
        ).create(),
    )
    # 创建 ChromaStore 对象，用于管理矢量存储
    return ChromaStore(config)


def _create_temporary_connection():
    """Create a temporary database connection for testing."""
    # 调用 SQLiteTempConnector 类的方法创建临时 SQLite 数据库连接
    connect = SQLiteTempConnector.create_temporary_db()
    # 创建临时表格并插入数据，连接对象的方法被调用
    connect.create_temp_tables(
        {
            # 第一个临时表格名为"user"
            "user": {
                # 指定表格列的结构，包括"id"（整数型主键）和"name"（文本类型）以及"age"（整数型）
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "name": "TEXT",
                    "age": "INTEGER",
                },
                # 插入具体的数据行：(1, "Tom", 8), (2, "Jerry", 16), ...
                "data": [
                    (1, "Tom", 8),
                    (2, "Jerry", 16),
                    (3, "Jack", 18),
                    (4, "Alice", 20),
                    (5, "Bob", 22),
                ],
            }
        }
    )
    
    connect.create_temp_tables(
        {
            # 第二个临时表格名为"job"
            "job": {
                # 指定表格列的结构，包括"id"（整数型主键）和"name"（文本类型）以及"age"（整数型）
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "name": "TEXT",
                    "age": "INTEGER",
                },
                # 插入具体的数据行：(1, "student", 8), (2, "student", 16), ...
                "data": [
                    (1, "student", 8),
                    (2, "student", 16),
                    (3, "student", 18),
                    (4, "teacher", 20),
                    (5, "teacher", 22),
                ],
            }
        }
    )
    
    connect.create_temp_tables(
        {
            # 第三个临时表格名为"student"
            "student": {
                # 指定表格列的结构，包括"id"（整数型主键）、"name"（文本类型）、"age"（整数型）和"info"（文本类型）
                "columns": {
                    "id": "INTEGER PRIMARY KEY",
                    "name": "TEXT",
                    "age": "INTEGER",
                    "info": "TEXT",
                },
                # 插入具体的数据行：(1, "Andy", 8, "good"), (2, "Jerry", 16, "bad"), ...
                "data": [
                    (1, "Andy", 8, "good"),
                    (2, "Jerry", 16, "bad"),
                    (3, "Wendy", 18, "good"),
                    (4, "Spider", 20, "bad"),
                    (5, "David", 22, "bad"),
                ],
            }
        }
    )
    # 返回连接对象
    return connect
def _prompt_join_fn(query: str, chunks: str) -> str:
    # 根据输入的查询和指令片段生成提示语句
    prompt = INSTRUCTION.format(chunks + INPUT_PROMPT.format(query))
    return prompt


class TriggerReqBody(BaseModel):
    query: str = Field(..., description="User query")


class RequestHandleOperator(MapOperator[TriggerReqBody, Dict]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, input_value: TriggerReqBody) -> Dict:
        # 将输入值封装成参数字典
        params = {
            "query": input_value.query,
        }
        print(f"Receive input value: {input_value.query}")
        return params


class SqlGenOperator(MapOperator[Any, Any]):
    """The Sql Generation Operator."""

    def __init__(self, llm: Optional[LLMClient], model_name: str, **kwargs):
        """Init the sql generation operator
        Args:
           llm (Optional[LLMClient]): base llm
           model_name (str): name of the model
        """
        super().__init__(**kwargs)
        self._llm = llm
        self._model_name = model_name

    async def map(self, prompt_with_query_and_schema: str) -> str:
        """generate sql by llm.
        Args:
            prompt_with_query_and_schema (str): prompt
        Return:
            str: sql
        """
        # 将输入的提示语句封装成消息列表
        messages = [
            ModelMessage(
                role=ModelMessageRoleType.SYSTEM, content=prompt_with_query_and_schema
            )
        ]
        # 构建模型请求
        request = ModelRequest(model=self._model_name, messages=messages)
        # 异步生成 SQL
        tasks = [self._llm.generate(request)]
        output = await run_async_tasks(tasks=tasks, concurrency_limit=1)
        sql = output[0].text
        return sql


class SqlExecOperator(MapOperator[Any, Any]):
    """The Sql Execution Operator."""

    def __init__(self, connector: Optional[RDBMSConnector] = None, **kwargs):
        """
        Args:
            connector (Optional[RDBMSConnector]): RDBMSConnector connection
        """
        super().__init__(**kwargs)
        self._connector = connector

    def map(self, sql: str) -> DataFrame:
        """retrieve table schemas.
        Args:
            sql (str): query.
        Return:
            DataFrame: sql execution result
        """
        # 执行 SQL 并将结果封装成 DataFrame
        dataframe = self._connector.run_to_df(command=sql, fetch="all")
        print(f"sql data is \n{dataframe}")
        return dataframe


class ChartDrawOperator(MapOperator[Any, Any]):
    """The Chart Draw Operator."""

    def __init__(self, **kwargs):
        """
        Args:
        connection (RDBMSConnector): The connection.
        """
        super().__init__(**kwargs)

    def map(self, df: DataFrame) -> str:
        """get sql result in db and draw.
        Args:
            df (DataFrame): DataFrame containing SQL result.
        """
        import matplotlib.pyplot as plt

        # 提取类别列和计数列
        category_column = df.columns[0]
        count_column = df.columns[1]
        # 绘制柱状图
        plt.figure(figsize=(8, 4))
        plt.bar(df[category_column], df[count_column])
        plt.xlabel(category_column)
        plt.ylabel(count_column)
        plt.show()
        return str(df)
# 创建一个 Directed Acyclic Graph (DAG)，命名为 "simple_nl_schema_sql_chart_example"
with DAG("simple_nl_schema_sql_chart_example") as dag:
    # 创建一个 HTTP 触发器对象，指定路径和请求方法为 POST，请求体为 TriggerReqBody
    trigger = HttpTrigger(
        "/examples/rag/schema_linking", methods="POST", request_body=TriggerReqBody
    )
    # 创建请求处理操作符对象
    request_handle_task = RequestHandleOperator()
    # 创建映射操作符对象，使用 lambda 函数提取请求中的 "query" 字段
    query_operator = MapOperator(lambda request: request["query"])
    # 创建 OpenAI LLM 客户端对象
    llm = OpenAILLMClient()
    # 指定模型名称为 "gpt-3.5-turbo"
    model_name = "gpt-3.5-turbo"
    # 创建模式链接操作符对象，使用临时连接器和指定的 LLM 客户端及模型名称
    retriever_task = SchemaLinkingOperator(
        connector=_create_temporary_connection(), llm=llm, model_name=model_name
    )
    # 创建连接操作符对象，指定合并函数为 _prompt_join_fn
    prompt_join_operator = JoinOperator(combine_function=_prompt_join_fn)
    # 创建 SQL 生成操作符对象，使用指定的 LLM 客户端和模型名称
    sql_gen_operator = SqlGenOperator(llm=llm, model_name=model_name)
    # 创建 SQL 执行操作符对象，使用临时连接器
    sql_exec_operator = SqlExecOperator(connector=_create_temporary_connection())
    # 创建绘图操作符对象，使用临时连接器
    draw_chart_operator = ChartDrawOperator(connector=_create_temporary_connection())
    
    # 设置任务依赖关系
    trigger >> request_handle_task >> query_operator >> prompt_join_operator
    trigger >> request_handle_task >> query_operator >> retriever_task >> prompt_join_operator
    prompt_join_operator >> sql_gen_operator >> sql_exec_operator >> draw_chart_operator

# 当脚本作为主程序运行时执行以下代码块
if __name__ == "__main__":
    # 如果 DAG 的第一个叶节点处于开发模式
    if dag.leaf_nodes[0].dev_mode:
        # 开发模式下，可以在本地调试运行 DAG
        from dbgpt.core.awel import setup_dev_environment
        
        # 设置开发环境，传入 DAG 列表和端口号 5555
        setup_dev_environment([dag], port=5555)
    else:
        # 否则，不做任何操作
        pass
```