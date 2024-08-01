# `.\DB-GPT-src\dbgpt\agent\resource\database.py`

```py
"""
Database resource module.
"""

import dataclasses             # 导入用于数据类的模块
import logging                 # 导入日志模块
from concurrent.futures import Executor, ThreadPoolExecutor  # 导入并发执行和线程池执行器
from typing import TYPE_CHECKING, Any, Generic, List, Optional, Tuple, Union  # 导入类型提示相关模块

import cachetools             # 导入缓存工具

from dbgpt.util.cache_utils import cached       # 导入缓存装饰器
from dbgpt.util.executor_utils import blocking_func_to_async   # 导入将阻塞函数转换为异步的工具函数

from .base import P, Resource, ResourceParameters, ResourceType  # 从本地相对路径导入基础类和类型定义

if TYPE_CHECKING:
    from dbgpt.datasource.rdbms.base import RDBMSConnector   # 导入类型检查时需要的特定模块

logger = logging.getLogger(__name__)   # 获取当前模块的日志记录器实例

_DEFAULT_PROMPT_TEMPLATE = (
    "Database type: {db_type}, related table structure definition: {schemas}"
)
_DEFAULT_PROMPT_TEMPLATE_ZH = "数据库类型：{db_type}，相关表结构定义：{schemas}"


@dataclasses.dataclass   # 数据类装饰器，用于声明数据类
class DBParameters(ResourceParameters):
    """DB parameters class."""
    
    db_name: str = dataclasses.field(metadata={"help": "DB name"})   # 数据类字段：数据库名称


class DBResource(Resource[P], Generic[P]):
    """Database resource class."""
    
    def __init__(
        self,
        name: str,
        db_type: Optional[str] = None,
        db_name: Optional[str] = None,
        dialect: Optional[str] = None,
        executor: Optional[Executor] = None,
        prompt_template: str = _DEFAULT_PROMPT_TEMPLATE,
    ):
        """Initialize the DB resource."""
        self._name = name                    # 设置资源名称属性
        self._db_type = db_type              # 设置数据库类型属性
        self._db_name = db_name              # 设置数据库名称属性
        self._dialect = dialect or db_type   # 设置方言属性，默认为数据库类型
        self._executor = executor or ThreadPoolExecutor()   # 设置执行器属性，默认为线程池执行器
        self._prompt_template = prompt_template   # 设置提示模板属性

    @classmethod
    def type(cls) -> ResourceType:
        """Return the resource type."""
        return ResourceType.DB    # 返回资源类型为数据库类型

    @property
    def name(self) -> str:
        """Return the resource name."""
        return self._name    # 返回资源的名称属性值

    @property
    def db_type(self) -> str:
        """Return the resource name."""
        if not self._db_type:
            raise ValueError("Database type is not set.")
        return self._db_type   # 返回数据库类型属性值

    @property
    def dialect(self) -> str:
        """Return the resource name."""
        if not self._dialect:
            raise ValueError("Dialect is not set.")
        return self._dialect   # 返回方言属性值

    @cached(cachetools.TTLCache(maxsize=100, ttl=10))   # 使用缓存装饰器，设置缓存过期时间和最大容量
    async def get_prompt(
        self,
        *,
        lang: str = "en",
        prompt_type: str = "default",
        question: Optional[str] = None,
        resource_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Get the prompt."""
        if not self._db_name:
            return "No database name provided."   # 如果未设置数据库名称，返回相应提示信息
        schema_info = await blocking_func_to_async(
            self._executor, self.get_schema_link, db=self._db_name, question=question
        )   # 使用异步方式获取数据库架构信息
        return self._prompt_template.format(db_type=self._db_type, schemas=schema_info)   # 返回格式化后的提示信息
    # 执行资源操作
class RDBMSConnectorResource(DBResource[DBParameters]):
    """Connector resource class."""

    def __init__(
        self,
        name: str,
        connector: Optional["RDBMSConnector"] = None,
        db_name: Optional[str] = None,
        db_type: Optional[str] = None,
        dialect: Optional[str] = None,
        executor: Optional[Executor] = None,
        **kwargs,
    ):
        """Initialize the connector resource."""
        # 如果没有指定 db_type 但提供了 connector 对象，则使用 connector 的 db_type
        if not db_type and connector:
            db_type = connector.db_type
        # 如果没有指定 dialect 但提供了 connector 对象，则使用 connector 的 dialect
        if not dialect and connector:
            dialect = connector.dialect
        # 如果没有指定 db_name 但提供了 connector 对象，则获取当前数据库名称作为 db_name
        if not db_name and connector:
            db_name = connector.get_current_db_name()
        # 设置内部的 connector 属性
        self._connector = connector
        # 调用父类的初始化方法，传递参数 name, db_type, db_name, dialect, executor, **kwargs
        super().__init__(
            name,
            db_type=db_type,
            db_name=db_name,
            dialect=dialect,
            executor=executor,
            **kwargs,
        )

    @property
    def connector(self) -> "RDBMSConnector":
        """Return the connector."""
        # 如果内部的 connector 属性未设置，则抛出 ValueError 异常
        if not self._connector:
            raise ValueError("Connector is not set.")
        # 返回内部的 connector 属性
        return self._connector

    def get_schema_link(
        self, db: str, question: Optional[str] = None
    ) -> Union[str, List[str]]:
        """Return the schema link of the database."""
        # 导入 _parse_db_summary 函数
        from dbgpt.rag.summary.rdbms_db_summary import _parse_db_summary

        # 调用 _parse_db_summary 函数，传入 connector 对象作为参数
        return _parse_db_summary(self.connector)

    def _sync_query(self, db: str, sql: str) -> Tuple[Tuple, List]:
        """Return the query result."""
        # 调用 connector 对象的 run 方法执行 SQL 查询，返回结果列表 result_lst
        result_lst = self.connector.run(sql)
        # 第一个元素为列名列表 columns
        columns = result_lst[0]
        # 后续元素为行数据列表 values
        values = result_lst[1:]
        # 返回列名列表和值列表作为元组
        return columns, values


class SQLiteDBResource(RDBMSConnectorResource):
    """SQLite database resource class."""

    def __init__(
        self, name: str, db_name: str, executor: Optional[Executor] = None, **kwargs
    ):
        """Initialize the SQLite database resource."""
        # 导入 SQLiteConnector 类
        from dbgpt.datasource.rdbms.conn_sqlite import SQLiteConnector

        # 使用 SQLiteConnector 类的 from_file_path 方法创建连接对象 conn
        conn = SQLiteConnector.from_file_path(db_name)
        # 调用父类 RDBMSConnectorResource 的初始化方法，传入参数 name, conn, executor, **kwargs
        super().__init__(name, conn, executor=executor, **kwargs)
```