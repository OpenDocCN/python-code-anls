# `.\DB-GPT-src\examples\sdk\chat_data_with_awel.py`

```py
# 异步编程库的导入
import asyncio
# JSON 数据处理库的导入
import json
# 文件和目录操作库的导入
import shutil

# 数据处理和分析库的导入
import pandas as pd

# 自定义配置文件的路径导入
from dbgpt.configs.model_config import PILOT_PATH
# 模板文件和输出解析器的导入
from dbgpt.core import (
    ChatPromptTemplate,
    HumanPromptTemplate,
    SQLOutputParser,
    SystemPromptTemplate,
)
# 数据处理流程相关类的导入
from dbgpt.core.awel import (
    DAG,
    BranchOperator,
    InputOperator,
    InputSource,
    JoinOperator,
    MapOperator,
    is_empty_data,
)
# 自定义操作符的导入
from dbgpt.core.operators import PromptBuilderOperator, RequestBuilderOperator
# 数据源操作相关类的导入
from dbgpt.datasource.operators import DatasourceOperator
# SQLite 临时连接器的导入
from dbgpt.datasource.rdbms.conn_sqlite import SQLiteTempConnector
# 模型操作相关类的导入
from dbgpt.model.operators import LLMOperator
# OpenAI LLM 客户端代理的导入
from dbgpt.model.proxy import OpenAILLMClient
# RAG 模块相关参数和操作符的导入
from dbgpt.rag import ChunkParameters
from dbgpt.rag.embedding import DefaultEmbeddingFactory
from dbgpt.rag.operators import DBSchemaAssemblerOperator, DBSchemaRetrieverOperator
# 向量存储模块相关类的导入
from dbgpt.storage.vector_store.chroma_store import ChromaStore, ChromaVectorConfig

# 删除旧的向量存储目录(/tmp/awel_with_data_vector_store)
shutil.rmtree("/tmp/awel_with_data_vector_store", ignore_errors=True)

# 打开 OpenAI 的默认嵌入工厂
embeddings = DefaultEmbeddingFactory.openai()

# 使用 OpenAI 的 LLM 模型客户端
llm_client = OpenAILLMClient()

# 创建 SQLite 临时数据库连接
db_conn = SQLiteTempConnector.create_temporary_db()
# 创建临时数据库表格 "user"，并插入初始数据
db_conn.create_temp_tables(
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

# 配置 Chroma 向量存储的参数
config = ChromaVectorConfig(
    persist_path=PILOT_PATH,
    name="db_schema_vector_store",
    embedding_fn=embeddings,
)
# 创建 Chroma 向量存储实例
vector_store = ChromaStore(config)

# 以下是关于 antv_charts 的一组注释，描述每种图表的用途和特点
antv_charts = [
    {"response_line_chart": "用于显示比较趋势分析数据"},
    {
        "response_pie_chart": "适合于比例和分布统计等场景"
    },
    {
        "response_table": "适合于显示有多个显示列或非数值列的情况"
    },
    # {"response_data_text":" 默认的显示方法，适合于单行或简单内容的显示"},
    {
        "response_scatter_plot": "适合于探索变量之间的关系，检测异常值等"
    },
    {
        "response_bubble_chart": "适合于多变量之间的关系，突出异常值或特殊情况等"
    },
    {
        "response_donut_chart": "适合于层次结构表示、类别比例显示和突出重要类别等"
    },
    {
        "response_area_chart": "适合于时间序列数据可视化、多组数据比较、数据变化趋势分析等"
    },
]
    {
        "response_heatmap": "Suitable for visual analysis of time series data, large-scale data sets, distribution of classified data, etc."
    },
    
    
    
    # 创建一个包含 "response_heatmap" 键的字典，其值是描述此热图类型适用性的字符串
]
# 此处有一个语法错误，缺少了开始方括号 `[`，需要修正语法错误

display_type = "\n".join(
    f"{key}:{value}" for dict_item in antv_charts for key, value in dict_item.items()
)
# 根据 antv_charts 中的字典项生成格式化的字符串列表，每行为 key:value 的形式，并用换行符连接起来，存储在 display_type 变量中

system_prompt = """You are a database expert. Please answer the user's question based on the database selected by the user and some of the available table structure definitions of the database.
Database name:
    {db_name}
Table structure definition:
    {table_info}

Constraint:
1.Please understand the user's intention based on the user's question, and use the given table structure definition to create a grammatically correct {dialect} sql. If sql is not required, answer the user's question directly.. 
2.Always limit the query to a maximum of {top_k} results unless the user specifies in the question the specific number of rows of data he wishes to obtain.
3.You can only use the tables provided in the table structure information to generate sql. If you cannot generate sql based on the provided table structure, please say: "The table structure information provided is not enough to generate sql queries." It is prohibited to fabricate information at will.
4.Please be careful not to mistake the relationship between tables and columns when generating SQL.
5.Please check the correctness of the SQL and ensure that the query performance is optimized under correct conditions.
6.Please choose the best one from the display methods given below for data rendering, and put the type name into the name parameter value that returns the required format. If you cannot find the most suitable one, use 'Table' as the display method.
the available data display methods are as follows: {display_type}

User Question:
    {user_input}
Please think step by step and respond according to the following JSON format:
    {response}
Ensure the response is correct json and can be parsed by Python json.loads.
"""
# 定义了一个包含多行文字的 system_prompt 字符串，用于系统提示和用户输入问题的格式化回应

RESPONSE_FORMAT_SIMPLE = {
    "thoughts": "thoughts summary to say to user",
    "sql": "SQL Query to run",
    "display_type": "Data display method",
}
# 定义了一个简单的 JSON 格式 RESPONSE_FORMAT_SIMPLE，包含了系统回应的结构信息，包括 thoughts, sql 和 display_type 字段

prompt = ChatPromptTemplate(
    messages=[
        SystemPromptTemplate.from_template(
            system_prompt,
            response_format=json.dumps(
                RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4
            ),
        ),
        HumanPromptTemplate.from_template("{user_input}"),
    ]
)
# 定义了一个 ChatPromptTemplate 对象 prompt，包含了系统提示和用户输入模板，其中系统提示基于 system_prompt 和 RESPONSE_FORMAT_SIMPLE 生成

class TwoSumOperator(MapOperator[pd.DataFrame, int]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def map(self, df: pd.DataFrame) -> int:
        return await self.blocking_func_to_async(self._two_sum, df)

    def _two_sum(self, df: pd.DataFrame) -> int:
        return df["age"].sum()
# 定义了一个 TwoSumOperator 类，继承自 MapOperator，实现了对 DataFrame 中 "age" 列求和的功能

def branch_even(x: int) -> bool:
    return x % 2 == 0
# 定义了一个函数 branch_even，用于判断给定整数 x 是否为偶数，返回布尔值

def branch_odd(x: int) -> bool:
    return not branch_even(x)
# 定义了一个函数 branch_odd，用于判断给定整数 x 是否为奇数，返回布尔值

class DataDecisionOperator(BranchOperator[int, int]):
    def __init__(self, odd_task_name: str, even_task_name: str, **kwargs):
        super().__init__(**kwargs)
        self.odd_task_name = odd_task_name
        self.even_task_name = even_task_name
# 定义了一个 DataDecisionOperator 类，继承自 BranchOperator，用于根据输入整数是奇数还是偶数执行不同的任务
    # 定义一个异步方法 branches，返回一个字典，其中包含偶数分支和奇数分支对应的任务名称
    async def branches(self):
        # 返回一个字典，键为 branch_even，值为 self.even_task_name；键为 branch_odd，值为 self.odd_task_name
        return {branch_even: self.even_task_name, branch_odd: self.odd_task_name}
with DAG("load_schema_dag") as load_schema_dag:
    # 创建一个虚拟输入任务
    input_task = InputOperator.dummy_input()
    
    # 加载数据库模式到向量存储
    assembler_task = DBSchemaAssemblerOperator(
        connector=db_conn,
        index_store=vector_store,
        chunk_parameters=ChunkParameters(chunk_strategy="CHUNK_BY_SIZE"),
    )
    # 将输入任务的输出连接到装配器任务
    input_task >> assembler_task

# 运行装配器任务并获取结果
chunks = asyncio.run(assembler_task.call())
print(chunks)


with DAG("chat_data_dag") as chat_data_dag:
    # 创建一个从可调用对象获取输入的输入任务
    input_task = InputOperator(input_source=InputSource.from_callable())
    
    # 数据库模式检索器任务
    retriever_task = DBSchemaRetrieverOperator(
        top_k=1,
        index_store=vector_store,
    )
    
    # 提取内容任务，从每个块对象中提取内容
    content_task = MapOperator(lambda cks: [c.content for c in cks])
    
    # 合并任务，将表信息和额外字典合并为单个字典
    merge_task = JoinOperator(
        lambda table_info, ext_dict: {"table_info": table_info, **ext_dict}
    )
    
    # 构建提示任务，创建查询提示
    prompt_task = PromptBuilderOperator(prompt)
    
    # 请求构建任务，使用指定模型构建请求
    req_build_task = RequestBuilderOperator(model="gpt-3.5-turbo")
    
    # LLM操作任务，使用LLM客户端执行语言模型操作
    llm_task = LLMOperator(llm_client=llm_client)
    
    # SQL输出解析器任务，解析SQL输出
    sql_parse_task = SQLOutputParser()
    
    # 数据源操作任务，执行与数据库的数据交互
    db_query_task = DatasourceOperator(connector=db_conn)

    # 构建任务流水线，依次连接任务
    (
        input_task
        >> MapOperator(lambda x: x["user_input"])  # 映射操作，提取用户输入
        >> retriever_task  # 检索器任务，从数据库检索信息
        >> content_task  # 内容任务，提取块的内容
        >> merge_task  # 合并任务，合并表信息和额外字典
    )
    # 连接输入任务到合并任务
    input_task >> merge_task
    
    # 连接合并任务到提示构建任务，请求构建任务，LLM任务和SQL解析任务
    merge_task >> prompt_task >> req_build_task >> llm_task >> sql_parse_task
    
    # 连接SQL解析任务到映射操作任务，然后连接到数据库查询任务
    sql_parse_task >> MapOperator(lambda x: x["sql"]) >> db_query_task

    # 两数求和操作任务
    two_sum_task = TwoSumOperator()
    
    # 数据决策操作任务，指定奇数和偶数任务名
    decision_task = DataDecisionOperator(
        odd_task_name="odd_task", even_task_name="even_task"
    )
    
    # 奇数操作任务，处理奇数情况
    odd_task = OddOperator(task_name="odd_task")
    
    # 偶数操作任务，处理偶数情况
    even_task = EvenOperator(task_name="even_task")
    
    # 合并操作任务，将奇数和偶数处理结果合并
    merge_task = MergeOperator()

    # 连接数据库查询任务到两数求和任务，然后连接到数据决策任务
    db_query_task >> two_sum_task >> decision_task
    
    # 连接数据决策任务到奇数操作任务，然后连接到合并操作任务
    decision_task >> odd_task >> merge_task
    
    # 连接数据决策任务到偶数操作任务，然后连接到合并操作任务
    decision_task >> even_task >> merge_task

# 运行合并操作任务并传入参数字典，获取最终结果
final_result = asyncio.run(
    merge_task.call(
        {
            "user_input": "Query the name and age of users younger than 18 years old",
            "db_name": "user_management",
            "dialect": "SQLite",
            "top_k": 1,
            "display_type": display_type,
            "response": json.dumps(
                RESPONSE_FORMAT_SIMPLE, ensure_ascii=False, indent=4
            ),
        }
    )
)
# 打印最终结果
print("The final result is:")
# 打印变量 final_result 的内容到标准输出
print(final_result)
```