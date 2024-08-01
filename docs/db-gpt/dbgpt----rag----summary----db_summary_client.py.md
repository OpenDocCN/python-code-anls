# `.\DB-GPT-src\dbgpt\rag\summary\db_summary_client.py`

```py
# 导入必要的模块和库
import logging  # 导入日志记录模块
import traceback  # 导入异常追踪模块

# 从内部配置中导入必要的类和常量
from dbgpt._private.config import Config
from dbgpt.component import SystemApp
from dbgpt.configs.model_config import EMBEDDING_MODEL_CONFIG
from dbgpt.rag.summary.gdbms_db_summary import GdbmsSummary
from dbgpt.rag.summary.rdbms_db_summary import RdbmsSummary

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 从配置中获取全局配置对象
CFG = Config()

# 定义一个用于处理数据库摘要的客户端类
class DBSummaryClient:
    """The client for DBSummary.

    DB Summary client, provide db_summary_embedding(put db profile and table profile
    summary into vector store), get_similar_tables method(get user query related tables
    info)

    Args:
        system_app (SystemApp): Main System Application class that manages the
            lifecycle and registration of components..
    """

    def __init__(self, system_app: SystemApp):
        """Create a new DBSummaryClient."""
        # 初始化函数，接收一个 SystemApp 实例作为参数
        self.system_app = system_app

        # 从 embedding_factory 模块中导入 EmbeddingFactory 类
        from dbgpt.rag.embedding.embedding_factory import EmbeddingFactory

        # 使用系统应用实例获取 embedding_factory 组件，并创建一个 EmbeddingFactory 实例
        embedding_factory: EmbeddingFactory = self.system_app.get_component(
            "embedding_factory", component_type=EmbeddingFactory
        )
        
        # 根据配置选择对应的嵌入模型名称，创建 embeddings 属性
        self.embeddings = embedding_factory.create(
            model_name=EMBEDDING_MODEL_CONFIG[CFG.EMBEDDING_MODEL]
        )

    def db_summary_embedding(self, dbname, db_type):
        """Put db profile and table profile summary into vector store."""
        # 创建数据库摘要客户端对象
        db_summary_client = self.create_summary_client(dbname, db_type)

        # 初始化数据库概要文件
        self.init_db_profile(db_summary_client, dbname)

        # 记录日志，指示数据库摘要嵌入操作成功
        logger.info("db summary embedding success")

    def get_db_summary(self, dbname, query, topk):
        """Get user query related tables info."""
        # 从 dbgpt.serve.rag.connector 模块导入 VectorStoreConnector 类
        from dbgpt.serve.rag.connector import VectorStoreConnector
        # 从 dbgpt.storage.vector_store.base 模块导入 VectorStoreConfig 类
        from dbgpt.storage.vector_store.base import VectorStoreConfig

        # 创建向量存储配置对象
        vector_store_config = VectorStoreConfig(name=dbname + "_profile")
        # 使用默认方式创建向量存储连接器
        vector_connector = VectorStoreConnector.from_default(
            CFG.VECTOR_STORE_TYPE,
            embedding_fn=self.embeddings,
            vector_store_config=vector_store_config,
        )

        # 从 dbgpt.rag.retriever.db_schema 模块导入 DBSchemaRetriever 类
        from dbgpt.rag.retriever.db_schema import DBSchemaRetriever

        # 创建数据库模式检索器实例，指定检索的顶部 K 个结果和向量存储的索引客户端
        retriever = DBSchemaRetriever(
            top_k=topk, index_store=vector_connector.index_client
        )

        # 使用检索器检索与用户查询相关的表信息
        table_docs = retriever.retrieve(query)
        # 提取表文档中的内容，并存储在列表中返回
        ans = [d.content for d in table_docs]
        return ans

    def init_db_summary(self):
        """Initialize db summary profile."""
        # 从配置中获取本地数据库管理器对象
        db_mange = CFG.local_db_manager
        # 获取数据库列表
        dbs = db_mange.get_db_list()
        # 遍历数据库列表
        for item in dbs:
            try:
                # 尝试执行数据库摘要嵌入操作
                self.db_summary_embedding(item["db_name"], item["db_type"])
            except Exception as e:
                # 捕获异常并记录警告日志
                message = traceback.format_exc()
                logger.warn(
                    f'{item["db_name"]}, {item["db_type"]} summary error!{str(e)}, '
                    f"detail: {message}"
                )
    def init_db_profile(self, db_summary_client, dbname):
        """Initialize db summary profile.

        Args:
        db_summary_client(DBSummaryClient): DB Summary Client  # 接收一个数据库摘要客户端对象
        dbname(str): dbname  # 数据库名称

        """
        vector_store_name = dbname + "_profile"  # 构建向量存储的名称，使用数据库名称加上"_profile"
        from dbgpt.serve.rag.connector import VectorStoreConnector  # 导入向量存储连接器
        from dbgpt.storage.vector_store.base import VectorStoreConfig  # 导入向量存储配置类

        vector_store_config = VectorStoreConfig(name=vector_store_name)  # 创建向量存储的配置对象，使用指定名称
        vector_connector = VectorStoreConnector.from_default(
            CFG.VECTOR_STORE_TYPE,  # 使用全局配置中的向量存储类型
            self.embeddings,  # 使用当前对象的嵌入数据
            vector_store_config=vector_store_config,  # 指定向量存储的配置对象
        )
        if not vector_connector.vector_name_exists():  # 检查向量存储是否存在指定名称的向量
            from dbgpt.rag.assembler.db_schema import DBSchemaAssembler  # 导入数据库模式装配器

            db_assembler = DBSchemaAssembler.load_from_connection(
                connector=db_summary_client.db,  # 使用数据库摘要客户端的数据库连接
                index_store=vector_connector.index_client,  # 使用向量连接器的索引客户端
            )

            if len(db_assembler.get_chunks()) > 0:  # 检查数据库装配器中是否有数据块
                db_assembler.persist()  # 持久化数据库装配器的状态
        else:
            logger.info(f"Vector store name {vector_store_name} exist")  # 记录日志，指定名称的向量存储已存在
        logger.info("initialize db summary profile success...")  # 记录日志，初始化数据库摘要配置成功

    def delete_db_profile(self, dbname):
        """Delete db profile.

        Args:
        dbname (str): The name of the database to delete.
        """
        vector_store_name = dbname + "_profile"  # 构建要删除的向量存储的名称
        from dbgpt.serve.rag.connector import VectorStoreConnector  # 导入向量存储连接器
        from dbgpt.storage.vector_store.base import VectorStoreConfig  # 导入向量存储配置类

        vector_store_config = VectorStoreConfig(name=vector_store_name)  # 创建向量存储的配置对象，使用指定名称
        vector_connector = VectorStoreConnector.from_default(
            CFG.VECTOR_STORE_TYPE,  # 使用全局配置中的向量存储类型
            self.embeddings,  # 使用当前对象的嵌入数据
            vector_store_config=vector_store_config,  # 指定向量存储的配置对象
        )
        vector_connector.delete_vector_name(vector_store_name)  # 删除指定名称的向量存储
        logger.info(f"delete db profile {dbname} success")  # 记录日志，删除指定数据库摘要配置成功

    @staticmethod
    def create_summary_client(dbname: str, db_type: str):
        """
        Create a summary client based on the database type.

        Args:
            dbname (str): The name of the database.
            db_type (str): The type of the database.

        Returns:
            SummaryClient: A summary client instance based on the database type.
        """
        if "graph" in db_type:  # 如果数据库类型包含"graph"
            return GdbmsSummary(dbname, db_type)  # 返回图数据库摘要客户端实例
        else:
            return RdbmsSummary(dbname, db_type)  # 返回关系型数据库摘要客户端实例
```