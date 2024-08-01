# `.\DB-GPT-src\dbgpt\app\base.py`

```py
import logging
import os
import signal
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional

from dbgpt._private.config import Config
from dbgpt.component import SystemApp
from dbgpt.storage import DBType
from dbgpt.util.parameter_utils import BaseServerParameters

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

logger = logging.getLogger(__name__)


def signal_handler(sig, frame):
    # 当接收到信号时，打印信息并调用 os._exit(0) 退出程序
    print("in order to avoid chroma db atexit problem")
    os._exit(0)


def async_db_summary(system_app: SystemApp):
    """async db schema into vector db"""
    from dbgpt.rag.summary.db_summary_client import DBSummaryClient

    client = DBSummaryClient(system_app=system_app)
    # 创建线程来异步初始化数据库摘要
    thread = threading.Thread(target=client.init_db_summary)
    thread.start()


def server_init(param: "WebServerParameters", system_app: SystemApp):
    # logger.info(f"args: {args}")
    # 初始化配置对象
    cfg = Config()
    # 将系统应用对象赋值给配置对象的 SYSTEM_APP 属性
    cfg.SYSTEM_APP = system_app
    # 初始化数据库存储
    _initialize_db_storage(param, system_app)

    # 注册信号处理函数，捕获 SIGINT 信号并调用 signal_handler 函数处理
    signal.signal(signal.SIGINT, signal_handler)


def _create_model_start_listener(system_app: SystemApp):
    def startup_event(wh):
        # 打印启动事件信息
        print("begin run _add_app_startup_event")
        # 异步执行数据库摘要生成函数
        async_db_summary(system_app)

    return startup_event


def _initialize_db_storage(param: "WebServerParameters", system_app: SystemApp):
    """Initialize the db storage.

    Now just support sqlite and mysql. If db type is sqlite, the db path is `pilot/meta_data/{db_name}.db`.
    """
    # 根据参数初始化数据库存储
    _initialize_db(
        try_to_create_db=not param.disable_alembic_upgrade, system_app=system_app
    )


def _migration_db_storage(param: "WebServerParameters"):
    """Migration the db storage."""
    # 导入所有模型以确保它们已经在 SQLAlchemy 中注册
    from dbgpt.app.initialization.db_model_initialization import _MODELS
    from dbgpt.configs.model_config import PILOT_PATH

    # 设置默认的元数据路径
    default_meta_data_path = os.path.join(PILOT_PATH, "meta_data")
    # 如果未禁用 alembic 升级，则执行以下操作
    if not param.disable_alembic_upgrade:
        # 导入数据库管理模块和数据库迁移工具函数
        from dbgpt.storage.metadata.db_manager import db
        from dbgpt.util._db_migration_utils import _ddl_init_and_upgrade

        # 尝试创建所有数据表，当数据库类型为 sqlite 时，将自动创建和升级系统模式
        # 否则，需要执行初始化脚本来创建模式
        CFG = Config()
        if CFG.LOCAL_DB_TYPE == "sqlite":
            try:
                # 创建所有数据表
                db.create_all()
            except Exception as e:
                # 记录创建表格时可能出现的异常
                logger.warning(
                    f"Create all tables stored in this metadata error: {str(e)}"
                )

            # 执行数据库初始化和升级操作
            _ddl_init_and_upgrade(default_meta_data_path, param.disable_alembic_upgrade)
        else:
            # 对于安全考虑，MySQL 数据库不支持DDL初始化和升级
            warn_msg = """For safety considerations, MySQL Database not support DDL init and upgrade. "
                "1.If you are use DB-GPT firstly, please manually execute the following command to initialize, 
                `mysql -h127.0.0.1 -uroot -p{your_password} < ./assets/schema/dbgpt.sql` "
                "2.If there are any changes to the table columns in the DB-GPT database, 
                it is necessary to compare with the DB-GPT/assets/schema/dbgpt.sql file 
                and manually make the columns changes in the MySQL database instance."""
            # 记录警告消息
            logger.warning(warn_msg)
def _initialize_db(
    try_to_create_db: Optional[bool] = False, system_app: Optional[SystemApp] = None
) -> str:
    """Initialize the database

    Now just support sqlite and MySQL. If db type is sqlite, the db path is `pilot/meta_data/{db_name}.db`.
    """
    # 导入必要的模块和类
    from urllib.parse import quote
    from urllib.parse import quote_plus as urlquote

    from dbgpt.configs.model_config import PILOT_PATH
    from dbgpt.datasource.rdbms.dialect.oceanbase.ob_dialect import (
        OBDialect,
    )  # noqa: F401
    from dbgpt.storage.metadata.db_manager import initialize_db

    # 获取配置信息
    CFG = Config()
    db_name = CFG.LOCAL_DB_NAME
    default_meta_data_path = os.path.join(PILOT_PATH, "meta_data")
    # 创建默认的元数据目录，如果目录不存在则创建
    os.makedirs(default_meta_data_path, exist_ok=True)

    # 根据配置的数据库类型选择相应的连接方式和路径
    if CFG.LOCAL_DB_TYPE == DBType.MySQL.value():
        db_url = (
            f"mysql+pymysql://{quote(CFG.LOCAL_DB_USER)}:"
            f"{urlquote(CFG.LOCAL_DB_PASSWORD)}@"
            f"{CFG.LOCAL_DB_HOST}:"
            f"{str(CFG.LOCAL_DB_PORT)}/"
            f"{db_name}?charset=utf8mb4"
        )
        # 尝试创建数据库，如果失败则抛出异常
        _create_mysql_database(db_name, db_url, try_to_create_db)
    elif CFG.LOCAL_DB_TYPE == DBType.OceanBase.value():
        db_url = (
            f"mysql+ob://{quote(CFG.LOCAL_DB_USER)}:"
            f"{urlquote(CFG.LOCAL_DB_PASSWORD)}@"
            f"{CFG.LOCAL_DB_HOST}:"
            f"{str(CFG.LOCAL_DB_PORT)}/"
            f"{db_name}?charset=utf8mb4"
        )
        _create_mysql_database(db_name, db_url, try_to_create_db)
    else:
        # 如果数据库类型未知，默认使用 SQLite，并设置路径
        sqlite_db_path = os.path.join(default_meta_data_path, f"{db_name}.db")
        db_url = f"sqlite:///{sqlite_db_path}"

    # 配置数据库引擎参数
    engine_args = {
        "pool_size": CFG.LOCAL_DB_POOL_SIZE,
        "max_overflow": CFG.LOCAL_DB_POOL_OVERFLOW,
        "pool_timeout": 30,
        "pool_recycle": 3600,
        "pool_pre_ping": True,
    }
    # 初始化数据库连接
    db = initialize_db(db_url, db_name, engine_args)

    # 如果有系统应用对象，则注册统一的数据库管理工厂
    if system_app:
        from dbgpt.storage.metadata import UnifiedDBManagerFactory

        system_app.register(UnifiedDBManagerFactory, db)

    # 返回默认的元数据路径
    return default_meta_data_path


def _create_mysql_database(db_name: str, db_url: str, try_to_create_db: bool = False):
    """Create mysql database if not exists

    Args:
        db_name (str): The database name
        db_url (str): The database url, include host, port, user, password and database name
        try_to_create_db (bool, optional): Whether to try to create database. Defaults to False.

    Raises:
        Exception: Raise exception if database operation failed
    """
    # 导入必要的模块和类
    from sqlalchemy import DDL, create_engine
    from sqlalchemy.exc import OperationalError, SQLAlchemyError

    # 如果不需要尝试创建数据库，则记录日志并返回
    if not try_to_create_db:
        logger.info(f"Skipping creation of database {db_name}")
        return

    # 使用 SQLAlchemy 创建数据库引擎
    engine = create_engine(db_url)

    # 省略了数据库创建的具体操作，将在实际使用中执行
    # 尝试连接到数据库
    try:
        # 使用引擎对象连接数据库
        with engine.connect() as conn:
            # 记录日志，说明数据库已存在
            logger.info(f"Database {db_name} already exists")
            # 如果连接成功，直接返回，不执行后续操作
            return
    except OperationalError as oe:
        # 捕获数据库操作错误
        # 如果错误提示数据库不存在，则尝试创建数据库
        if "Unknown database" in str(oe):
            try:
                # 从数据库连接 URL 中移除数据库名称部分
                no_db_name_url = db_url.rsplit("/", 1)[0]
                # 创建不含数据库名称的引擎对象
                engine_no_db = create_engine(no_db_name_url)
                with engine_no_db.connect() as conn:
                    # 执行 SQL 语句，创建指定名称的数据库
                    conn.execute(
                        DDL(
                            f"CREATE DATABASE {db_name} CHARACTER SET utf8mb4 COLLATE "
                            f"utf8mb4_unicode_ci"
                        )
                    )
                    # 记录日志，说明数据库创建成功
                    logger.info(f"Database {db_name} successfully created")
            except SQLAlchemyError as e:
                # 记录错误日志，说明数据库创建失败
                logger.error(f"Failed to create database {db_name}: {e}")
                # 抛出异常，终止程序执行
                raise
        else:
            # 记录错误日志，说明连接数据库时发生其他错误
            logger.error(f"Error connecting to database {db_name}: {oe}")
            # 抛出异常，终止程序执行
            raise
@dataclass
class WebServerParameters(BaseServerParameters):
    # 定义Web服务器的参数类，继承自BaseServerParameters类

    host: Optional[str] = field(
        default="0.0.0.0", metadata={"help": "Webserver deploy host"}
    )
    # Web服务器部署的主机地址，默认为"0.0.0.0"

    port: Optional[int] = field(
        default=None, metadata={"help": "Webserver deploy port"}
    )
    # Web服务器部署的端口号，默认为None

    daemon: Optional[bool] = field(
        default=False, metadata={"help": "Run Webserver in background"}
    )
    # 是否在后台运行Web服务器，默认为False

    controller_addr: Optional[str] = field(
        default=None,
        metadata={
            "help": "The Model controller address to connect. If None, read model "
            "controller address from environment key `MODEL_SERVER`."
        },
    )
    # 模型控制器的连接地址，如果为None，则从环境变量`MODEL_SERVER`中读取

    model_name: str = field(
        default=None,
        metadata={
            "help": "The default model name to use. If None, read model name from "
            "environment key `LLM_MODEL`.",
            "tags": "fixed",
        },
    )
    # 默认要使用的模型名称，如果为None，则从环境变量`LLM_MODEL`中读取

    share: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to create a publicly shareable link for the interface. "
            "Creates an SSH tunnel to make your UI accessible from anywhere. "
        },
    )
    # 是否创建一个公开可分享的界面链接，通过SSH隧道使UI在任何地方都可以访问

    remote_embedding: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enable remote embedding models. If it is True, you need"
            " to start a embedding model through `dbgpt start worker --worker_type "
            "text2vec --model_name xxx --model_path xxx`"
        },
    )
    # 是否启用远程嵌入模型，如果为True，则需要通过命令启动一个嵌入模型

    remote_rerank: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to enable remote rerank models. If it is True, you need"
            " to start a rerank model through `dbgpt start worker --worker_type "
            "text2vec --rerank --model_name xxx --model_path xxx`"
        },
    )
    # 是否启用远程重新排序模型，如果为True，则需要通过命令启动一个重新排序模型

    light: Optional[bool] = field(default=False, metadata={"help": "enable light mode"})
    # 是否启用轻量模式，默认为False

    log_file: Optional[str] = field(
        default="dbgpt_webserver.log",
        metadata={
            "help": "The filename to store log",
        },
    )
    # 存储日志的文件名，默认为"dbgpt_webserver.log"

    tracer_file: Optional[str] = field(
        default="dbgpt_webserver_tracer.jsonl",
        metadata={
            "help": "The filename to store tracer span records",
        },
    )
    # 存储跟踪器跨度记录的文件名，默认为"dbgpt_webserver_tracer.jsonl"

    tracer_storage_cls: Optional[str] = field(
        default=None,
        metadata={
            "help": "The storage class to storage tracer span records",
        },
    )
    # 存储跟踪器跨度记录的存储类，默认为None

    disable_alembic_upgrade: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to disable alembic to initialize and upgrade database metadata",
        },
    )
    # 是否禁用alembic来初始化和升级数据库元数据，默认为False

    awel_dirs: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directories to search awel files, split by `,`",
        },
    )
    # 搜索awel文件的目录列表，以逗号分隔，默认为None
    default_thread_pool_size: Optional[int] = field(
        # 定义一个可选的整数类型变量 default_thread_pool_size，初始值为 None
        default=None,
        # 元数据字段，用于存储帮助信息
        metadata={
            "help": "The default thread pool size, If None, "
                    "use default config of python thread pool",
        },
    )
```