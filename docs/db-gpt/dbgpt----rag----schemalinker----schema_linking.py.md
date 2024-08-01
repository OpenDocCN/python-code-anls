# `.\DB-GPT-src\dbgpt\rag\schemalinker\schema_linking.py`

```py
"""SchemaLinking by LLM."""

# 导入所需模块和类
from functools import reduce  # 导入 reduce 函数，用于将列表中的元素累加
from typing import List, Optional, cast  # 引入类型提示，包括列表、可选类型和类型转换

# 导入核心类和函数
from dbgpt.core import (
    Chunk,  # 导入 Chunk 类，用于表示文本块
    LLMClient,  # 导入 LLMClient 类，用于与模型通信
    ModelMessage,  # 导入 ModelMessage 类，表示模型的消息
    ModelMessageRoleType,  # 导入 ModelMessageRoleType 类，表示消息的角色类型
    ModelRequest,  # 导入 ModelRequest 类，表示模型的请求
)
# 导入数据源连接基类
from dbgpt.datasource.base import BaseConnector
# 导入索引存储基类
from dbgpt.rag.index.base import IndexStoreBase
# 导入基础模式链接器
from dbgpt.rag.schemalinker.base_linker import BaseSchemaLinker
# 导入关系数据库摘要解析函数
from dbgpt.rag.summary.rdbms_db_summary import _parse_db_summary
# 导入异步任务运行函数
from dbgpt.util.chat_util import run_async_tasks

# 定义指导信息
INSTRUCTION = """
You need to filter out the most relevant database table schema information (it may be a
 single table or multiple tables) required to generate the SQL of the question query
 from the given database schema information. First, I will show you an example of an
 instruction followed by the correct schema response. Then, I will give you a new
 instruction, and you should write the schema response that appropriately completes the
 request.

### Example1 Instruction:
['job(id, name, age)', 'user(id, name, age)', 'student(id, name, age, info)']
### Example1 Input:
Find the age of student table
### Example1 Response:
['student(id, name, age, info)']
###New Instruction:
{}
"""

# 定义输入提示信息
INPUT_PROMPT = "\n###New Input:\n{}\n###New Response:"


class SchemaLinking(BaseSchemaLinker):
    """SchemaLinking by LLM."""

    def __init__(
        self,
        connector: BaseConnector,
        model_name: str,
        llm: LLMClient,
        top_k: int = 5,
        index_store: Optional[IndexStoreBase] = None,
    ):
        """Create the schema linking instance.

        Args:
           connection (Optional[BaseConnector]): BaseConnector connection.
           llm (Optional[LLMClient]): base llm
        """
        self._top_k = top_k  # 设置查询结果的前 k 个最相关结果
        self._connector = connector  # 设置数据源连接器
        self._llm = llm  # 设置 LLN 客户端实例
        self._model_name = model_name  # 设置模型名称
        self._index_store = index_store  # 设置索引存储实例

    def _schema_linking(self, query: str) -> List:
        """Get all db schema info."""
        # 解析数据库摘要信息并封装为 Chunk 对象列表
        table_summaries = _parse_db_summary(self._connector)
        chunks = [Chunk(content=table_summary) for table_summary in table_summaries]
        chunks_content = [chunk.content for chunk in chunks]
        return chunks_content

    def _schema_linking_with_vector_db(self, query: str) -> List[Chunk]:
        """Get db schema info with vector database."""
        queries = [query]
        # 检查是否提供了索引存储实例，若未提供则引发 ValueError 异常
        if not self._index_store:
            raise ValueError("Vector store connector is not provided.")
        # 查询每个查询的前 k 个相似结果
        candidates = [
            self._index_store.similar_search(query, self._top_k) for query in queries
        ]
        return cast(List[Chunk], reduce(lambda x, y: x + y, candidates))
    # 异步方法，用于将查询字符串链接到模式。返回一个包含连接内容的列表。
    async def _schema_linking_with_llm(self, query: str) -> List:
        # 使用schema_linking方法处理查询，返回处理后的内容
        chunks_content = self.schema_linking(query)
        # 构建包含查询内容的指令格式化字符串
        schema_prompt = INSTRUCTION.format(
            str(chunks_content) + INPUT_PROMPT.format(query)
        )
        # 创建包含系统消息的列表，消息内容为schema_prompt
        messages = [
            ModelMessage(role=ModelMessageRoleType.SYSTEM, content=schema_prompt)
        ]
        # 创建模型请求对象，指定模型名称和消息列表
        request = ModelRequest(model=self._model_name, messages=messages)
        # 生成异步任务列表，包含使用_llm生成请求的任务
        tasks = [self._llm.generate(request)]
        # 通过异步运行任务函数获取准确的模式信息，限制并发为1
        schema = await run_async_tasks(tasks=tasks, concurrency_limit=1)
        # 从schema结果中获取文本信息
        schema_text = schema[0].text
        # 返回获取的模式文本信息
        return schema_text
```