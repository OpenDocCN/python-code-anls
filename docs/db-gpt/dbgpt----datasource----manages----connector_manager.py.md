# `.\DB-GPT-src\dbgpt\datasource\manages\connector_manager.py`

```py
"""Connection manager."""
import logging
from typing import TYPE_CHECKING, List, Optional, Type

from dbgpt.component import BaseComponent, ComponentType, SystemApp
from dbgpt.storage.schema import DBType
from dbgpt.util.executor_utils import ExecutorFactory

from ..base import BaseConnector
from ..db_conn_info import DBConfig
from .connect_config_db import ConnectConfigDao

if TYPE_CHECKING:
    # 建议不要依赖 rag 模块。
    from dbgpt.rag.summary.db_summary_client import DBSummaryClient

logger = logging.getLogger(__name__)


class ConnectorManager(BaseComponent):
    """Connector manager."""

    name = ComponentType.CONNECTOR_MANAGER

    def __init__(self, system_app: SystemApp):
        """Create a new ConnectorManager."""
        # 初始化连接配置数据访问对象
        self.storage = ConnectConfigDao()
        # 设置系统应用
        self.system_app = system_app
        # 数据库摘要客户端，默认为 None
        self._db_summary_client: Optional["DBSummaryClient"] = None
        super().__init__(system_app)

    def init_app(self, system_app: SystemApp):
        """Init component."""
        # 初始化组件，设置系统应用
        self.system_app = system_app

    def on_init(self):
        """Execute on init.

        Load all connector classes.
        """
        # 导入所有连接器类，确保被导入以避免 linter 提示
        from dbgpt.datasource.conn_spark import SparkConnector  # noqa: F401
        from dbgpt.datasource.conn_tugraph import TuGraphConnector  # noqa: F401
        from dbgpt.datasource.rdbms.base import RDBMSConnector  # noqa: F401
        from dbgpt.datasource.rdbms.conn_clickhouse import (  # noqa: F401
            ClickhouseConnector,
        )
        from dbgpt.datasource.rdbms.conn_doris import DorisConnector  # noqa: F401
        from dbgpt.datasource.rdbms.conn_duckdb import DuckDbConnector  # noqa: F401
        from dbgpt.datasource.rdbms.conn_hive import HiveConnector  # noqa: F401
        from dbgpt.datasource.rdbms.conn_mssql import MSSQLConnector  # noqa: F401
        from dbgpt.datasource.rdbms.conn_mysql import MySQLConnector  # noqa: F401
        from dbgpt.datasource.rdbms.conn_oceanbase import OceanBaseConnect  # noqa: F401
        from dbgpt.datasource.rdbms.conn_postgresql import (  # noqa: F401
            PostgreSQLConnector,
        )
        from dbgpt.datasource.rdbms.conn_sqlite import SQLiteConnector  # noqa: F401
        from dbgpt.datasource.rdbms.conn_starrocks import (  # noqa: F401
            StarRocksConnector,
        )
        from dbgpt.datasource.rdbms.conn_vertica import VerticaConnector  # noqa: F401
        from dbgpt.datasource.rdbms.dialect.oceanbase.ob_dialect import (  # noqa: F401
            OBDialect,
        )

        from .connect_config_db import ConnectConfigEntity  # noqa: F401

    def before_start(self):
        """Execute before start."""
        # 初始化数据库摘要客户端
        from dbgpt.rag.summary.db_summary_client import DBSummaryClient

        self._db_summary_client = DBSummaryClient(self.system_app)

    @property
    def db_summary_client(self) -> "DBSummaryClient":
        """Get DBSummaryClient."""
        # 检查是否已经初始化了 DBSummaryClient，如果没有则抛出异常
        if not self._db_summary_client:
            raise ValueError("DBSummaryClient is not initialized")
        # 返回 DBSummaryClient 实例
        return self._db_summary_client

    def _get_all_subclasses(
        self, cls: Type[BaseConnector]
    ) -> List[Type[BaseConnector]]:
        """Get all subclasses of cls."""
        # 获取所有子类的列表
        subclasses = cls.__subclasses__()
        # 递归获取所有子类的子类
        for subclass in subclasses:
            subclasses += self._get_all_subclasses(subclass)
        return subclasses

    def get_all_completed_types(self) -> List[DBType]:
        """Get all completed types."""
        # 获取所有已完成的类型
        chat_classes = self._get_all_subclasses(BaseConnector)  # type: ignore
        support_types = []
        for cls in chat_classes:
            # 如果是正常类型，则添加到支持类型列表中
            if cls.db_type and cls.is_normal_type():
                db_type = DBType.of_db_type(cls.db_type)
                if db_type:
                    support_types.append(db_type)
        return support_types

    def get_cls_by_dbtype(self, db_type) -> Type[BaseConnector]:
        """Get class by db type."""
        # 获取指定数据库类型对应的类
        chat_classes = self._get_all_subclasses(BaseConnector)  # type: ignore
        result = None
        for cls in chat_classes:
            # 如果数据库类型匹配且是正常类型，则返回该类
            if cls.db_type == db_type and cls.is_normal_type():
                result = cls
        if not result:
            raise ValueError("Unsupported Db Type！" + db_type)
        return result

    def get_connector(self, db_name: str):
        """Create a new connection instance.

        Args:
            db_name (str): database name
        """
        # 获取数据库配置信息
        db_config = self.storage.get_db_config(db_name)
        # 获取数据库类型
        db_type = DBType.of_db_type(db_config.get("db_type"))
        if not db_type:
            raise ValueError("Unsupported Db Type！" + db_config.get("db_type"))
        # 根据数据库类型获取连接实例
        connect_instance = self.get_cls_by_dbtype(db_type.value())
        if db_type.is_file_db():
            # 如果是文件型数据库，则根据文件路径创建连接实例
            db_path = db_config.get("db_path")
            return connect_instance.from_file_path(db_path)  # type: ignore
        else:
            # 如果是 URI 型数据库，则根据连接信息创建连接实例
            db_host = db_config.get("db_host")
            db_port = db_config.get("db_port")
            db_user = db_config.get("db_user")
            db_pwd = db_config.get("db_pwd")
            return connect_instance.from_uri_db(  # type: ignore
                host=db_host, port=db_port, user=db_user, pwd=db_pwd, db_name=db_name
            )
    # 测试数据库连接是否可用，并返回连接器实例
    def test_connect(self, db_info: DBConfig) -> BaseConnector:
        """Test connectivity.

        Args:
            db_info (DBConfig): db connect info.

        Returns:
            BaseConnector: connector instance.

        Raises:
            ValueError: Test connect Failure.
        """
        try:
            # 获取数据库类型的枚举值
            db_type = DBType.of_db_type(db_info.db_type)
            if not db_type:
                # 如果数据库类型不支持，则抛出异常
                raise ValueError("Unsupported Db Type！" + db_info.db_type)
            
            # 根据数据库类型获取连接器类实例
            connect_instance = self.get_cls_by_dbtype(db_type.value())
            
            if db_type.is_file_db():
                # 如果是文件型数据库，使用文件路径创建连接器实例
                db_path = db_info.file_path
                return connect_instance.from_file_path(db_path)  # type: ignore
            else:
                # 如果是URI型数据库，使用URI信息创建连接器实例
                db_name = db_info.db_name
                db_host = db_info.db_host
                db_port = db_info.db_port
                db_user = db_info.db_user
                db_pwd = db_info.db_pwd
                return connect_instance.from_uri_db(  # type: ignore
                    host=db_host,
                    port=db_port,
                    user=db_user,
                    pwd=db_pwd,
                    db_name=db_name,
                )
        except Exception as e:
            # 捕获所有异常，并记录错误日志
            logger.error(f"{db_info.db_name} Test connect Failure!{str(e)}")
            # 抛出自定义异常，说明连接测试失败
            raise ValueError(f"{db_info.db_name} Test connect Failure!{str(e)}")

    # 获取数据库列表
    def get_db_list(self):
        """Get db list."""
        return self.storage.get_db_list()

    # 删除指定名称的数据库连接信息
    def delete_db(self, db_name: str):
        """Delete db connect info."""
        return self.storage.delete_db(db_name)

    # 编辑数据库连接信息
    def edit_db(self, db_info: DBConfig):
        """Edit db connect info."""
        return self.storage.update_db_info(
            db_info.db_name,
            db_info.db_type,
            db_info.file_path,
            db_info.db_host,
            db_info.db_port,
            db_info.db_user,
            db_info.db_pwd,
            db_info.comment,
        )

    # 异步生成数据库摘要信息的嵌入
    async def async_db_summary_embedding(self, db_name, db_type):
        """Async db summary embedding."""
        # 获取默认执行器的实例，并创建任务提交给执行器
        executor = self.system_app.get_component(
            ComponentType.EXECUTOR_DEFAULT, ExecutorFactory
        ).create()  # type: ignore
        executor.submit(self.db_summary_client.db_summary_embedding, db_name, db_type)
        # 返回异步操作成功的标志
        return True
    def add_db(self, db_info: DBConfig):
        """Add db connect info.

        Args:
            db_info (DBConfig): db connect info.
        """
        # 记录日志，输出添加的数据库连接信息
        logger.info(f"add_db:{db_info.__dict__}")
        try:
            # 解析数据库类型
            db_type = DBType.of_db_type(db_info.db_type)
            # 如果数据库类型不支持，抛出异常
            if not db_type:
                raise ValueError("Unsupported Db Type！" + db_info.db_type)
            # 如果是文件类型数据库，添加文件类型数据库连接
            if db_type.is_file_db():
                self.storage.add_file_db(
                    db_info.db_name, db_info.db_type, db_info.file_path
                )
            else:
                # 否则，添加 URL 类型数据库连接
                self.storage.add_url_db(
                    db_info.db_name,
                    db_info.db_type,
                    db_info.db_host,
                    db_info.db_port,
                    db_info.db_user,
                    db_info.db_pwd,
                    db_info.comment,
                )
            # 异步执行嵌入操作
            executor = self.system_app.get_component(
                ComponentType.EXECUTOR_DEFAULT, ExecutorFactory
            ).create()  # type: ignore
            # 提交数据库摘要嵌入任务
            executor.submit(
                self.db_summary_client.db_summary_embedding,
                db_info.db_name,
                db_info.db_type,
            )
        except Exception as e:
            # 捕获异常并抛出具体错误信息
            raise ValueError("Add db connect info error!" + str(e))

        # 操作成功完成，返回 True 表示添加成功
        return True
```