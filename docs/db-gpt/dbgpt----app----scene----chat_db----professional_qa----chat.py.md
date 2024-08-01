# `.\DB-GPT-src\dbgpt\app\scene\chat_db\professional_qa\chat.py`

```py
from typing import Dict
from dbgpt._private.config import Config
from dbgpt.app.scene import BaseChat, ChatScene
from dbgpt.util.executor_utils import blocking_func_to_async
from dbgpt.util.tracer import trace

CFG = Config()

class ChatWithDbQA(BaseChat):
    chat_scene: str = ChatScene.ChatWithDbQA.value()

    keep_end_rounds = 5

    """As a DBA, Chat DB Module, chat with combine DB meta schema """

    def __init__(self, chat_param: Dict):
        """Chat DB Module Initialization
        
        Args:
            - chat_param: Dict
                - chat_session_id: (str) chat session_id
                - current_user_input: (str) current user input
                - model_name:(str) llm model name
                - select_param:(str) dbname
        """
        # Initialize the parent class with modified chat_param
        self.db_name = chat_param["select_param"]
        chat_param["chat_mode"] = ChatScene.ChatWithDbQA
        super().__init__(chat_param=chat_param)

        # Establish connection to the local database if dbname is provided
        if self.db_name:
            self.database = CFG.local_db_manager.get_connector(self.db_name)
            self.tables = self.database.get_table_names()
        
        # Determine top_k based on database type (graph or non-graph)
        if self.database.is_graph_type():
            # Calculate top_k based on the sum of vertex and edge tables
            self.top_k = len(self.tables["vertex_tables"]) + len(self.tables["edge_tables"])
        else:
            # Print the database type if not graph type
            print(self.database.db_type)
            # Set top_k to either CFG.KNOWLEDGE_SEARCH_TOP_SIZE or number of tables
            self.top_k = (CFG.KNOWLEDGE_SEARCH_TOP_SIZE if len(self.tables) > CFG.KNOWLEDGE_SEARCH_TOP_SIZE else len(self.tables))

    @trace()
    async def generate_input_values(self) -> Dict:
        # 初始化一个空字符串用于存储表信息
        table_info = ""
        # 设置数据库方言为 MySQL
        dialect = "mysql"
        try:
            # 尝试导入 DBSummaryClient 类
            from dbgpt.rag.summary.db_summary_client import DBSummaryClient
        except ImportError:
            # 如果导入失败，抛出值错误异常
            raise ValueError("Could not import DBSummaryClient. ")

        if self.db_name:
            # 如果有指定数据库名，则创建一个 DBSummaryClient 的实例
            client = DBSummaryClient(system_app=CFG.SYSTEM_APP)
            try:
                # 尝试获取数据库摘要信息，将执行结果赋值给 table_infos
                # 注意：以下代码段为异步调用，使用了非阻塞函数转换成异步的方式
                table_infos = await blocking_func_to_async(
                    self._executor,
                    client.get_db_summary,
                    self.db_name,
                    self.current_user_input,
                    self.top_k,
                )
            except Exception as e:
                # 如果获取摘要信息出错，打印异常信息
                print("db summary find error!" + str(e))
                # 出错时，获取简单的表信息
                table_infos = await blocking_func_to_async(
                    self._executor, self.database.table_simple_info
                )

            # 获取数据库的方言信息
            dialect = self.database.dialect

        # 构造输入值字典
        input_values = {
            "input": self.current_user_input,
            # 将 top_k 转换为字符串并加入输入值字典（已注释掉的部分）
            # "top_k": str(self.top_k),
            # 将方言信息加入输入值字典（已注释掉的部分）
            # "dialect": dialect,
            # 将表信息加入输入值字典
            "table_info": table_infos,
        }
        # 返回最终的输入值字典
        return input_values
```