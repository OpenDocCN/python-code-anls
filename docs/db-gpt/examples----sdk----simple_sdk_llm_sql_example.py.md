# `.\DB-GPT-src\examples\sdk\simple_sdk_llm_sql_example.py`

```py
import asyncio
import json  # 导入处理 JSON 数据的模块
from typing import Dict, List  # 引入类型提示，指定函数参数和返回值的类型

from dbgpt.core import SQLOutputParser  # 导入 SQL 输出解析器
from dbgpt.core.awel import (  # 导入 AWEL 框架的各种操作符和数据源
    DAG,  # 有向无环图数据结构
    InputOperator,  # 输入操作符
    JoinOperator,  # 连接操作符
    MapOperator,  # 映射操作符
    SimpleCallDataInputSource,  # 简单调用数据输入源
)
from dbgpt.core.operators import (  # 导入核心操作符
    BaseLLMOperator,  # 基础 LLM 操作符
    PromptBuilderOperator,  # 提示构建器操作符
    RequestBuilderOperator,  # 请求构建器操作符
)
from dbgpt.datasource.operators.datasource_operator import DatasourceOperator  # 导入数据源操作符
from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector  # 导入 SQLite 临时连接器
from dbgpt.model.proxy import OpenAILLMClient  # 导入 OpenAI LLM 客户端
from dbgpt.rag.operators.datasource import DatasourceRetrieverOperator  # 导入数据源检索操作符


def _create_temporary_connection():
    """创建一个用于测试的临时数据库连接。"""
    conn = SQLiteTempConnector.create_temporary_db()  # 创建临时 SQLite 数据库连接
    conn.create_temp_tables(
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
    )  # 在临时数据库中创建表结构和数据
    return conn  # 返回连接对象


def _sql_prompt() -> str:
    """这是一个用于生成 SQL 的提示模板。

    格式化参数：
        {db_name}: 数据库名称
        {table_info}: 表结构信息
        {dialect}: 数据库方言
        {top_k}: 最大结果数量
        {user_input}: 用户问题
        {response}: 响应格式

    返回：
        str: 提示模板字符串
    """
    return """请根据用户选择的数据库和数据库中部分可用的表结构信息回答用户的问题。 
数据库名称: 
    {db_name} 
    
表结构定义: 
    {table_info}

约束条件:
    1.请根据用户的问题理解用户的意图，并使用给定的表结构定义生成语法正确的 {dialect} SQL。如果不需要 SQL，请直接回答用户的问题。
    2.始终将查询限制为最多 {top_k} 条结果，除非用户在问题中指定希望获取的具体行数。
    3.只能使用表结构信息中提供的表来生成 SQL 查询。如果无法根据提供的表结构生成 SQL，请说：“提供的表结构信息不足以生成 SQL 查询。”严禁随意捏造信息。
    4.在生成 SQL 时请注意不要误解表和列之间的关系。
    5.请检查 SQL 的正确性，并确保在正确的条件下优化查询性能。

用户问题:
    {user_input}
请一步一步思考，并按照以下 JSON 格式进行响应:
    {response}
def _join_func(query_dict: Dict, db_summary: List[str]):
    """Join function for JoinOperator.

    Build the format arguments for the prompt template.

    Args:
        query_dict (Dict): The query dict from DAG input.
        db_summary (List[str]): The table structure information from DatasourceRetrieverOperator.

    Returns:
        Dict: The query dict with the format arguments.
    """
    # Define a default response dictionary structure
    default_response = {
        "thoughts": "thoughts summary to say to user",
        "sql": "SQL Query to run",
    }
    # Convert the default_response dictionary to a JSON string
    response = json.dumps(default_response, ensure_ascii=False, indent=4)
    # Assign table structure information to the query_dict
    query_dict["table_info"] = db_summary
    # Assign the JSON formatted response to the query_dict
    query_dict["response"] = response
    # Return the updated query_dict
    return query_dict


class SQLResultOperator(JoinOperator[Dict]):
    """Merge the SQL result and the model result."""

    def __init__(self, **kwargs):
        super().__init__(combine_function=self._combine_result, **kwargs)

    def _combine_result(self, sql_result_df, model_result: Dict) -> Dict:
        # Assign the SQL result dataframe to the model_result dictionary
        model_result["data_df"] = sql_result_df
        # Return the updated model_result dictionary
        return model_result


with DAG("simple_sdk_llm_sql_example") as dag:
    # Create a temporary database connection
    db_connection = _create_temporary_connection()
    # Define an input task using SimpleCallDataInputSource
    input_task = InputOperator(input_source=SimpleCallDataInputSource())
    # Define a retriever task to fetch datasource information
    retriever_task = DatasourceRetrieverOperator(connector=db_connection)
    
    # Merge input data and table structure information using JoinOperator
    prompt_input_task = JoinOperator(combine_function=_join_func)
    # Build prompts based on SQL queries
    prompt_task = PromptBuilderOperator(_sql_prompt())
    # Prepare requests for the model
    model_pre_handle_task = RequestBuilderOperator(model="gpt-3.5-turbo")
    # Initialize the BaseLLMOperator with OpenAILLMClient
    llm_task = BaseLLMOperator(OpenAILLMClient())
    # Parse SQL output
    out_parse_task = SQLOutputParser()
    # Map SQL queries
    sql_parse_task = MapOperator(map_function=lambda x: x["sql"])
    # Perform database queries
    db_query_task = DatasourceOperator(connector=db_connection)
    # Handle SQL result operations
    sql_result_task = SQLResultOperator()
    
    # Define task dependencies in the DAG
    input_task >> prompt_input_task
    input_task >> retriever_task >> prompt_input_task
    (
        prompt_input_task
        >> prompt_task
        >> model_pre_handle_task
        >> llm_task
        >> out_parse_task
        >> sql_parse_task
        >> db_query_task
        >> sql_result_task
    )
    out_parse_task >> sql_result_task


if __name__ == "__main__":
    # Define input data for the DAG execution
    input_data = {
        "db_name": "test_db",
        "dialect": "sqlite",
        "top_k": 5,
        "user_input": "What is the name and age of the user with age less than 18",
    }
    # Execute the SQL result task asynchronously with input data
    output = asyncio.run(sql_result_task.call(call_data=input_data))
    # Print the thoughts summary extracted from the output
    print(f"\nthoughts: {output.get('thoughts')}\n")
    # Print the SQL query extracted from the output
    print(f"sql: {output.get('sql')}\n")
    # Print the result data dataframe extracted from the output
    print(f"result data:\n{output.get('data_df')}")
```