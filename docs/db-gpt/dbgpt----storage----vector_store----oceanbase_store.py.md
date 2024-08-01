# `.\DB-GPT-src\dbgpt\storage\vector_store\oceanbase_store.py`

```py
"""OceanBase vector store."""
# 导入必要的模块和库
import json
import logging
import os
import threading
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# 导入 SQLAlchemy 相关模块
from pydantic import Field
from sqlalchemy import Column, Table, create_engine, insert, text
from sqlalchemy.dialects.mysql import JSON, LONGTEXT, VARCHAR
from sqlalchemy.types import String, UserDefinedType

# 导入 dbgpt 相关模块和类
from dbgpt.core import Chunk, Document, Embeddings
from dbgpt.core.awel.flow import Parameter, ResourceCategory, register_resource
from dbgpt.storage.vector_store.base import (
    _COMMON_PARAMETERS,
    VectorStoreBase,
    VectorStoreConfig,
)
from dbgpt.storage.vector_store.filters import MetadataFilters
from dbgpt.util.i18n_utils import _

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base

# 设置日志记录器
logger = logging.getLogger(__name__)
sql_logger = None
sql_dbg_log_path = os.getenv("OB_SQL_DBG_LOG_PATH", "")
if sql_dbg_log_path != "":
    # 如果定义了环境变量 OB_SQL_DBG_LOG_PATH，则创建 SQL 调试日志记录器
    sql_logger = logging.getLogger("ob_sql_dbg")
    sql_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(sql_dbg_log_path)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    sql_logger.addHandler(file_handler)

# 定义 OceanBase 默认参数常量
_OCEANBASE_DEFAULT_EMBEDDING_DIM = 1536
_OCEANBASE_DEFAULT_COLLECTION_NAME = "langchain_document"
_OCEANBASE_DEFAULT_IVFFLAT_ROW_THRESHOLD = 10000
_OCEANBASE_DEFAULT_RWLOCK_MAX_READER = 64

# 创建 SQLAlchemy 基类
Base = declarative_base()


def ob_vector_from_db(value):
    """Parse vector from oceanbase."""
    # 从 OceanBase 中解析向量数据并返回
    return [float(v) for v in value[1:-1].split(",")]


def ob_vector_to_db(value, dim=None):
    """Parse vector to oceanbase vector constant type."""
    # 将向量转换为 OceanBase 向量常量类型的字符串表示
    if value is None:
        return value

    return "[" + ",".join([str(float(v)) for v in value]) + "]"


class Vector(UserDefinedType):
    """OceanBase Vector Column Type."""

    cache_ok = True
    _string = String()

    def __init__(self, dim):
        """Create a Vector column with dimemsion `dim`."""
        super(UserDefinedType, self).__init__()
        self.dim = dim

    def get_col_spec(self, **kw):
        """Get vector column definition in string format."""
        # 返回向量列的字符串格式定义
        return "VECTOR(%d)" % self.dim

    def bind_processor(self, dialect):
        """Get a processor to parse an array to oceanbase vector."""

        def process(value):
            # 获取将数组解析为 OceanBase 向量的处理器
            return ob_vector_to_db(value, self.dim)

        return process

    def literal_processor(self, dialect):
        """Get a string processor to parse an array to OceanBase Vector."""
        # 获取将数组解析为 OceanBase 向量的字符串处理器
        string_literal_processor = self._string._cached_literal_processor(dialect)

        def process(value):
            return string_literal_processor(ob_vector_to_db(value, self.dim))

        return process
    def result_processor(self, dialect, coltype):
        """Get a processor to parse OceanBase Vector to array."""
        # 定义一个内部函数 process，用于将数据库中的 OceanBase Vector 转换为数组
        def process(value):
            # 调用 ob_vector_from_db 函数处理传入的 value，并返回处理后的结果
            return ob_vector_from_db(value)

        # 返回内部函数 process，作为结果处理器的处理函数
        return process
class OceanBaseCollectionStat:
    """A tracer that maintains a table status in OceanBase."""

    def __init__(self):
        """Create OceanBaseCollectionStat instance."""
        # 初始化对象，设置互斥锁和初始状态标志位
        self._lock = threading.Lock()
        self.maybe_collection_not_exist = True
        self.maybe_collection_index_not_exist = True

    def collection_exists(self):
        """Set a table is existing."""
        # 标记表存在，使用互斥锁保证线程安全
        with self._lock:
            self.maybe_collection_not_exist = False

    def collection_index_exists(self):
        """Set the index of a table is existing."""
        # 标记表索引存在，使用互斥锁保证线程安全
        with self._lock:
            self.maybe_collection_index_not_exist = False

    def collection_not_exists(self):
        """Set a table is dropped."""
        # 标记表被删除，使用互斥锁保证线程安全
        with self._lock:
            self.maybe_collection_not_exist = True

    def collection_index_not_exists(self):
        """Set the index of a table is dropped."""
        # 标记表索引被删除，使用互斥锁保证线程安全
        with self._lock:
            self.maybe_collection_index_not_exist = True

    def get_maybe_collection_not_exist(self):
        """Get the existing status of a table."""
        # 获取表存在状态，使用互斥锁保证读取操作的线程安全
        with self._lock:
            return self.maybe_collection_not_exist

    def get_maybe_collection_index_not_exist(self):
        """Get the existing stats of the index of a table."""
        # 获取表索引存在状态，使用互斥锁保证读取操作的线程安全
        with self._lock:
            return self.maybe_collection_index_not_exist


class OceanBaseGlobalRWLock:
    """A global rwlock for OceanBase to do creating vector index table offline ddl."""

    def __init__(self, max_readers) -> None:
        """Create a rwlock."""
        # 初始化读写锁对象
        self.max_readers_ = max_readers
        self.writer_entered_ = False
        self.reader_cnt_ = 0
        self.mutex_ = threading.Lock()
        self.writer_cv_ = threading.Condition(self.mutex_)
        self.reader_cv_ = threading.Condition(self.mutex_)

    def rlock(self):
        """Lock for reading."""
        # 获取读锁，处理并发读操作
        self.mutex_.acquire()
        while self.writer_entered_ or self.max_readers_ == self.reader_cnt_:
            self.reader_cv_.wait()
        self.reader_cnt_ += 1
        self.mutex_.release()

    def runlock(self):
        """Unlock reading lock."""
        # 释放读锁
        self.mutex_.acquire()
        self.reader_cnt_ -= 1
        if self.writer_entered_:
            if self.reader_cnt_ == 0:
                self.writer_cv_.notify(1)
        else:
            if self.max_readers_ - 1 == self.reader_cnt_:
                self.reader_cv_.notify(1)
        self.mutex_.release()

    def wlock(self):
        """Lock for writing."""
        # 获取写锁，处理并发写操作
        self.mutex_.acquire()
        while self.writer_entered_:
            self.reader_cv_.wait()
        self.writer_entered_ = True
        while 0 < self.reader_cnt_:
            self.writer_cv_.wait()
        self.mutex_.release()

    def wunlock(self):
        """Unlock writing lock."""
        # 释放写锁
        self.mutex_.acquire()
        self.writer_entered_ = False
        self.reader_cv_.notifyAll()
        self.mutex_.release()
    class OBRLock:
        """Reading Lock Wrapper for `with` clause."""

        def __init__(self, rwlock) -> None:
            """Create reading lock wrapper instance."""
            self.rwlock_ = rwlock

        def __enter__(self):
            """Lock."""
            # 获取读锁
            self.rwlock_.rlock()

        def __exit__(self, exc_type, exc_value, traceback):
            """Unlock when exiting."""
            # 释放读锁
            self.rwlock_.runlock()

    class OBWLock:
        """Writing Lock Wrapper for `with` clause."""

        def __init__(self, rwlock) -> None:
            """Create writing lock wrapper instance."""
            self.rwlock_ = rwlock

        def __enter__(self):
            """Lock."""
            # 获取写锁
            self.rwlock_.wlock()

        def __exit__(self, exc_type, exc_value, traceback):
            """Unlock when exiting."""
            # 释放写锁
            self.rwlock_.wunlock()

    def reader_lock(self):
        """Get reading lock wrapper."""
        # 返回一个读锁封装实例
        return self.OBRLock(self)

    def writer_lock(self):
        """Get writing lock wrapper."""
        # 返回一个写锁封装实例
        return self.OBWLock(self)
ob_grwlock = OceanBaseGlobalRWLock(_OCEANBASE_DEFAULT_RWLOCK_MAX_READER)
# 创建一个全局读写锁对象，使用默认的最大读者数量参数

class OceanBase:
    """OceanBase Vector Store implementation."""

    def __init__(
        self,
        database: str,
        connection_string: str,
        embedding_function: Embeddings,
        embedding_dimension: int = _OCEANBASE_DEFAULT_EMBEDDING_DIM,
        collection_name: str = _OCEANBASE_DEFAULT_COLLECTION_NAME,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        engine_args: Optional[dict] = None,
        delay_table_creation: bool = True,
        enable_index: bool = False,
        th_create_ivfflat_index: int = _OCEANBASE_DEFAULT_IVFFLAT_ROW_THRESHOLD,
        sql_logger: Optional[logging.Logger] = None,
        collection_stat: Optional[OceanBaseCollectionStat] = None,
        enable_normalize_vector: bool = False,
    ) -> None:
        """Create OceanBase Vector Store instance."""
        # 初始化 OceanBase 实例，设置各种属性
        self.database = database
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.embedding_dimension = embedding_dimension
        self.collection_name = collection_name
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.delay_table_creation = delay_table_creation
        self.th_create_ivfflat_index = th_create_ivfflat_index
        self.enable_index = enable_index
        self.sql_logger = sql_logger
        self.collection_stat = collection_stat
        self.enable_normalize_vector = enable_normalize_vector
        self.__post_init__(engine_args)

    def __post_init__(
        self,
        engine_args: Optional[dict] = None,
    ) -> None:
        """Create connection & vector table."""
        _engine_args = engine_args or {}
        # 如果 _engine_args 中没有设置 'pool_recycle' 参数，则设置默认值为 3600
        if "pool_recycle" not in _engine_args:
            _engine_args["pool_recycle"] = 3600
        # 创建数据库连接引擎，使用给定的连接字符串和引擎参数
        self.engine = create_engine(self.connection_string, **_engine_args)
        # 创建向量表
        self.create_collection()

    @property
    def embeddings(self) -> Embeddings:
        """Get embedding function."""
        # 返回嵌入函数对象
        return self.embedding_function

    def create_collection(self) -> None:
        """Create vector table."""
        # 如果需要预先删除向量表，则执行删除操作
        if self.pre_delete_collection:
            self.delete_collection()
        # 如果不延迟表的创建，并且集合统计对象为 None 或者可能不存在集合
        if not self.delay_table_creation and (
            self.collection_stat is None
            or self.collection_stat.get_maybe_collection_not_exist()
        ):
            # 创建表（如果表不存在的话）
            self.create_table_if_not_exists()
            # 如果集合统计对象不为 None，则标记集合已存在
            if self.collection_stat is not None:
                self.collection_stat.collection_exists()
    def delete_collection(self) -> None:
        """删除向量表."""
        # 构建删除表的SQL语句
        drop_statement = text(f"DROP TABLE IF EXISTS {self.collection_name}")
        # 如果存在SQL日志器，记录删除表操作
        if self.sql_logger is not None:
            self.sql_logger.debug(f"Trying to delete collection: {drop_statement}")
        # 使用数据库引擎连接，执行删除表操作
        with self.engine.connect() as conn, conn.begin():
            conn.execute(drop_statement)
            # 如果存在集合状态对象，通知集合不存在
            if self.collection_stat is not None:
                self.collection_stat.collection_not_exists()
                # 如果存在集合状态对象，通知集合索引不存在
                self.collection_stat.collection_index_not_exists()

    def create_table_if_not_exists(self) -> None:
        """使用SQL创建向量表."""
        # 构建创建表的SQL查询语句
        create_table_query = f"""
            CREATE TABLE IF NOT EXISTS `{self.collection_name}` (
                id VARCHAR(40) NOT NULL,
                embedding VECTOR({self.embedding_dimension}),
                document LONGTEXT,
                metadata JSON,
                PRIMARY KEY (id)
            )
        """
        # 如果存在SQL日志器，记录创建表操作
        if self.sql_logger is not None:
            self.sql_logger.debug(f"Trying to create table: {create_table_query}")
        # 使用数据库引擎连接，执行创建表操作
        with self.engine.connect() as conn, conn.begin():
            # 创建表
            conn.execute(text(create_table_query))

    def create_collection_ivfflat_index_if_not_exists(self) -> None:
        """使用SQL创建ivfflat索引表."""
        # 构建创建ivfflat索引的SQL查询语句
        create_index_query = f"""
            CREATE INDEX IF NOT EXISTS `embedding_idx` on `{self.collection_name}` (
                embedding l2
            ) using ivfflat with (lists=20)
        """
        # 获取写锁并使用数据库引擎连接，执行创建索引表操作
        with ob_grwlock.writer_lock(), self.engine.connect() as conn, conn.begin():
            # 如果存在SQL日志器，记录创建ivfflat索引操作
            if self.sql_logger is not None:
                self.sql_logger.debug(
                    f"Trying to create ivfflat index: {create_index_query}"
                )
            # 执行创建索引操作
            conn.execute(text(create_index_query))

    def check_table_exists(self) -> bool:
        """检查表是否存在."""
        # 构建检查表是否存在的SQL查询语句
        check_table_query = f"""
            SELECT COUNT(*) as cnt
            FROM information_schema.tables
            WHERE table_schema='{self.database}' AND table_name='{self.collection_name}'
        """
        try:
            # 使用数据库引擎连接，执行检查表存在性操作，获取读锁
            with self.engine.connect() as conn, conn.begin(), ob_grwlock.reader_lock():
                # 执行SQL查询
                table_exists_res = conn.execute(text(check_table_query))
                for row in table_exists_res:
                    return row.cnt > 0
                # 如果没有cnt行，返回False以通过'make mypy'
                return False
        except Exception as e:
            # 记录错误日志
            logger.error(f"check_table_exists error: {e}")
            return False
    # 定义一个方法用于通过文本查询进行相似性搜索
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Do similarity search via query in String."""
        # 获取查询文本的嵌入向量
        embedding = self.embedding_function.embed_query(query)
        # 调用基于向量的相似性搜索方法，返回文档列表
        docs = self.similarity_search_by_vector(embedding=embedding, k=k, filter=filter)
        return docs

    # 定义一个方法用于通过向量进行相似性搜索
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[MetadataFilters] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Do similarity search via query vector."""
        # 调用基于向量和分数的相似性搜索方法，获取文档及其分数的列表
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter
        )
        # 返回文档列表，不包括分数信息
        return [doc for doc, _ in docs_and_scores]

    # 定义一个方法用于通过向量和分数进行相似性搜索
    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[MetadataFilters] = None,
        score_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        # 方法尚未完全实现，计划返回带有分数的文档列表
    ) -> List[Tuple[Document, float]]:
        """Do similarity search via query vector with score."""
        try:
            from sqlalchemy.engine import Row  # 导入 sqlalchemy 中的 Row 类
        except ImportError:
            raise ImportError(
                "Could not import Row from sqlalchemy.engine. "
                "Please 'pip install sqlalchemy>=1.4'."
            )

        # filter is not support in OceanBase currently.
        # 当前版本的 OceanBase 不支持 filter

        # normailze embedding vector
        if self.enable_normalize_vector:
            embedding = self._normalization_vectors(embedding)  # 如果启用向量归一化，则归一化查询向量 embedding

        embedding_str = ob_vector_to_db(embedding, self.embedding_dimension)  # 将归一化后的向量转换为数据库存储格式的字符串 embedding_str
        vector_distance_op = "<@>" if self.enable_normalize_vector else "<->"  # 根据是否启用向量归一化选择相似度计算操作符
        sql_query = f"""
            SELECT document, metadata, embedding {vector_distance_op} '{embedding_str}'
            as distance
            FROM {self.collection_name}
            ORDER BY embedding {vector_distance_op} '{embedding_str}'
            LIMIT :k
        """
        sql_query_str_for_log = f"""
            SELECT document, metadata, embedding {vector_distance_op} '?' as distance
            FROM {self.collection_name}
            ORDER BY embedding {vector_distance_op} '?'
            LIMIT {k}
        """

        params = {"k": k}  # SQL 查询参数，指定查询结果的数量上限为 k
        try:
            with ob_grwlock.reader_lock(), self.engine.connect() as conn:
                if self.sql_logger is not None:
                    self.sql_logger.debug(
                        f"Trying to do similarity search: {sql_query_str_for_log}"
                    )
                results: Sequence[Row] = conn.execute(
                    text(sql_query), params
                ).fetchall()  # 执行 SQL 查询并获取所有结果

            if (score_threshold is not None) and self.enable_normalize_vector:
                # 如果指定了 score_threshold 并且启用了向量归一化，则筛选结果中距离大于等于 score_threshold 的文档和分数
                documents_with_scores = [
                    (
                        Document(
                            content=result.document,
                            metadata=json.loads(result.metadata),
                        ),
                        result.distance,
                    )
                    for result in results
                    if result.distance >= score_threshold
                ]
            else:
                # 否则，返回所有查询结果的文档和分数
                documents_with_scores = [
                    (
                        Document(
                            content=result.document,
                            metadata=json.loads(result.metadata),
                        ),
                        result.distance,
                    )
                    for result in results
                ]
            return documents_with_scores  # 返回包含文档及其分数的列表
        except Exception as e:
            self.logger.error("similarity_search_with_score_by_vector failed:", str(e))  # 记录错误日志信息
            return []  # 返回空列表，表示查询失败或者没有符合条件的结果
    ) -> List[Tuple[Document, float]]:
        """Do similarity search via query String with score."""
        # Embed the query string to obtain its embedding vector
        embedding = self.embedding_function.embed_query(query)
        # Perform similarity search using the embedded query vector
        docs = self.similarity_search_with_score_by_vector(
            embedding=embedding, k=k, filter=filter, score_threshold=score_threshold
        )
        # Return the list of documents along with their similarity scores
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete vectors from vector table."""
        if ids is None:
            # Raise an error if no IDs are provided for deletion
            raise ValueError("No ids provided to delete.")

        # Define the table schema for the vector table
        chunks_table = Table(
            self.collection_name,
            Base.metadata,
            Column("id", VARCHAR(40), primary_key=True),
            Column("embedding", Vector(self.embedding_dimension)),
            Column("document", LONGTEXT, nullable=True),
            Column("metadata", JSON, nullable=True),  # Optional metadata field
            keep_existing=True,
        )

        try:
            with self.engine.connect() as conn, conn.begin():
                # Specify the condition for deletion based on provided IDs
                delete_condition = chunks_table.c.id.in_(ids)
                # Construct the SQL DELETE statement
                delete_stmt = chunks_table.delete().where(delete_condition)
                # Acquire a reader lock before executing the delete operation
                with ob_grwlock.reader_lock():
                    # Log the attempted delete operation for debugging purposes
                    if self.sql_logger is not None:
                        self.sql_logger.debug(
                            f"Trying to delete vectors: {str(delete_stmt)}"
                        )
                    # Execute the delete statement
                    conn.execute(delete_stmt)
                # Return True indicating successful deletion
                return True
        except Exception as e:
            # Log an error message if delete operation fails and return False
            self.logger.error("Delete operation failed:", str(e))
            return False

    def _normalization_vectors(self, vector):
        import numpy as np

        # Compute the L2 norm of the input vector
        norm = np.linalg.norm(vector)
        # Normalize the vector and convert it to a list
        return (vector / norm).tolist()

    @classmethod
    def connection_string_from_db_params(
        cls,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Get connection string."""
        # Construct and return a MySQL connection string based on provided parameters
        return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
# 创建一个线程锁，用于保护并发访问的对象集合统计信息
ob_collection_stats_lock = threading.Lock()

# 定义一个空的字典，用于存储OceanBaseCollectionStat对象，键为字符串类型
ob_collection_stats: Dict[str, OceanBaseCollectionStat] = {}


# 注册资源，名称为"OceanBase Vector Store"
# 类别为VECTOR_STORE，包含一些通用参数和特定的OceanBase参数
@register_resource(
    _("OceanBase Vector Store"),
    "oceanbase_vector_store",
    category=ResourceCategory.VECTOR_STORE,
    parameters=[
        *_COMMON_PARAMETERS,
        Parameter.build_from(
            _("OceanBase Host"),
            "ob_host",
            str,
            description=_("oceanbase host"),
            optional=True,
            default=None,
        ),
        Parameter.build_from(
            _("OceanBase Port"),
            "ob_port",
            int,
            description=_("oceanbase port"),
            optional=True,
            default=None,
        ),
        Parameter.build_from(
            _("OceanBase User"),
            "ob_user",
            str,
            description=_("user to login"),
            optional=True,
            default=None,
        ),
        Parameter.build_from(
            _("OceanBase Password"),
            "ob_password",
            str,
            description=_("password to login"),
            optional=True,
            default=None,
        ),
        Parameter.build_from(
            _("OceanBase Database"),
            "ob_database",
            str,
            description=_("database for vector tables"),
            optional=True,
            default=None,
        ),
    ],
    description="OceanBase vector store.",
)
# 定义OceanBaseConfig类，继承自VectorStoreConfig类，用于配置OceanBase的相关参数
class OceanBaseConfig(VectorStoreConfig):
    """OceanBase vector store config."""

    # 配置类，允许任意类型的字段
    class Config:
        """Config for BaseModel."""

        arbitrary_types_allowed = True

    """OceanBase config"""
    # OceanBase主机地址，默认为"127.0.0.1"
    ob_host: str = Field(
        default="127.0.0.1",
        description="oceanbase host",
    )
    # OceanBase端口号，默认为2881
    ob_port: int = Field(
        default=2881,
        description="oceanbase port",
    )
    # OceanBase登录用户名，默认为"root@test"
    ob_user: str = Field(
        default="root@test",
        description="user to login",
    )
    # OceanBase登录密码，默认为空字符串
    ob_password: str = Field(
        default="",
        description="password to login",
    )
    # OceanBase数据库名称，默认为"test"
    ob_database: str = Field(
        default="test",
        description="database for vector tables",
    )


# 定义OceanBaseStore类，继承自VectorStoreBase类，用于操作OceanBase向量存储
class OceanBaseStore(VectorStoreBase):
    """OceanBase vector store."""
    def __init__(self, vector_store_config: OceanBaseConfig) -> None:
        """Create a OceanBaseStore instance."""
        # 检查向量存储配置中是否提供了嵌入函数，如果未提供则引发数值错误
        if vector_store_config.embedding_fn is None:
            raise ValueError("embedding_fn is required for OceanBaseStore")
        # 调用父类的初始化方法
        super().__init__()
        # 从配置中获取嵌入函数并赋给实例变量
        self.embeddings = vector_store_config.embedding_fn
        # 设置集合名称
        self.collection_name = vector_store_config.name
        # 将配置转换为字典形式
        vector_store_config = vector_store_config.dict()
        # 设置 OceanBase 主机地址，优先使用配置中的值，其次从环境变量中获取，默认为 127.0.0.1
        self.OB_HOST = str(
            vector_store_config.get("ob_host") or os.getenv("OB_HOST", "127.0.0.1")
        )
        # 设置 OceanBase 端口，优先使用配置中的值，其次从环境变量中获取，默认为 2881
        self.OB_PORT = int(
            vector_store_config.get("ob_port") or int(os.getenv("OB_PORT", "2881"))
        )
        # 设置 OceanBase 用户名，优先使用配置中的值，其次从环境变量中获取，默认为 'root@test'
        self.OB_USER = str(
            vector_store_config.get("ob_user") or os.getenv("OB_USER", "root@test")
        )
        # 设置 OceanBase 密码，优先使用配置中的值，其次从环境变量中获取，默认为空字符串
        self.OB_PASSWORD = str(
            vector_store_config.get("ob_password") or os.getenv("OB_PASSWORD", "")
        )
        # 设置 OceanBase 数据库名称，优先使用配置中的值，其次从环境变量中获取，默认为 'test'
        self.OB_DATABASE = str(
            vector_store_config.get("ob_database") or os.getenv("OB_DATABASE", "test")
        )
        # 设置是否启用向量归一化，默认为从环境变量中获取的布尔值
        self.OB_ENABLE_NORMALIZE_VECTOR = bool(
            os.getenv("OB_ENABLE_NORMALIZE_VECTOR", "")
        )
        # 构建数据库连接字符串
        self.connection_string = OceanBase.connection_string_from_db_params(
            self.OB_HOST, self.OB_PORT, self.OB_DATABASE, self.OB_USER, self.OB_PASSWORD
        )
        # 设置日志记录器
        self.logger = logger
        # 使用锁确保线程安全地更新集合统计信息
        with ob_collection_stats_lock:
            # 如果当前集合的统计信息尚未初始化，则进行初始化
            if ob_collection_stats.get(self.collection_name) is None:
                ob_collection_stats[self.collection_name] = OceanBaseCollectionStat()
            # 获取当前集合的统计信息对象
            self.collection_stat = ob_collection_stats[self.collection_name]

        # 创建 OceanBase 客户端实例，用于与 OceanBase 交互
        self.vector_store_client = OceanBase(
            database=self.OB_DATABASE,
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            logger=self.logger,
            sql_logger=sql_logger,
            enable_index=bool(os.getenv("OB_ENABLE_INDEX", "")),
            collection_stat=self.collection_stat,
            enable_normalize_vector=self.OB_ENABLE_NORMALIZE_VECTOR,
        )

    def similar_search(
        self, text, topk, filters: Optional[MetadataFilters] = None, **kwargs: Any
    ) -> List[Chunk]:
        """Perform a search on a query string and return results."""
        # 记录搜索操作日志信息
        self.logger.info("OceanBase: similar_search..")
        # 使用向量存储客户端执行相似性搜索操作，获取文档列表
        documents = self.vector_store_client.similarity_search(
            text, topk, filter=filters
        )
        # 将搜索结果转换为 Chunk 对象列表并返回
        return [Chunk(content=doc.content, metadata=doc.metadata) for doc in documents]

    def similar_search_with_scores(
        self,
        text,
        topk,
        score_threshold: float,
        filters: Optional[MetadataFilters] = None,
        **kwargs: Any
    ) -> List[Chunk]:
        """Perform a search on a query string with score threshold and return results."""
        # 记录带分数阈值的相似性搜索操作日志信息
        self.logger.info("OceanBase: similar_search_with_scores..")
        # 使用向量存储客户端执行带分数阈值的相似性搜索操作，获取文档列表
        documents = self.vector_store_client.similarity_search(
            text, topk, score_threshold=score_threshold, filter=filters
        )
        # 将搜索结果转换为 Chunk 对象列表并返回
        return [Chunk(content=doc.content, metadata=doc.metadata) for doc in documents]
    ) -> List[Chunk]:
        """执行查询字符串的搜索并返回带有分数的结果。"""
        self.logger.info("OceanBase: similar_search_with_scores..")
        # 使用向量存储客户端执行带分数的相似性搜索
        docs_and_scores = self.vector_store_client.similarity_search_with_score(
            text, topk, filter=filters
        )
        # 构建结果列表，每个元素包含文档内容、元数据和分数
        return [
            Chunk(content=doc.content, metadata=doc.metadata, score=score)
            for doc, score in docs_and_scores
        ]

    def vector_name_exists(self):
        """检查向量名称是否存在。"""
        self.logger.info("OceanBase: vector_name_exists..")
        # 使用向量存储客户端检查表是否存在
        return self.vector_store_client.check_table_exists()

    def load_document(self, chunks: List[Chunk]) -> List[str]:
        """将文档加载到向量数据库中。"""
        self.logger.info("OceanBase: load_document..")
        # 提取每个 Chunk 对象的文本内容和元数据
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        # 使用向量存储客户端将文本和元数据添加到数据库，并返回添加的文档 ID 列表
        ids = self.vector_store_client.add_texts(texts=texts, metadatas=metadatas)
        return ids

    def delete_vector_name(self, vector_name):
        """删除向量名称。"""
        self.logger.info("OceanBase: delete_vector_name..")
        # 使用向量存储客户端删除集合（向量名称对应的数据集）
        return self.vector_store_client.delete_collection()

    def delete_by_ids(self, ids):
        """根据文档 ID 删除向量。"""
        self.logger.info("OceanBase: delete_by_ids..")
        # 将输入的 ID 字符串按逗号分隔为列表
        ids = ids.split(",")
        if len(ids) > 0:
            # 使用向量存储客户端根据 ID 列表删除对应的向量
            self.vector_store_client.delete(ids)
```