# `.\DB-GPT-src\dbgpt\util\_db_migration_utils.py`

```py
import logging  # 导入日志模块
import os  # 导入操作系统接口模块
from typing import Optional  # 导入类型提示模块

from alembic import command  # 导入 alembic 命令模块
from alembic.config import Config as AlembicConfig  # 导入 alembic 配置模块
from alembic.util.exc import CommandError  # 导入 alembic 异常模块
from sqlalchemy import Engine, text  # 导入 SQLAlchemy 引擎和文本模块
from sqlalchemy.orm import DeclarativeMeta, Session  # 导入 SQLAlchemy 声明元类和会话模块

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器对象


def create_alembic_config(
    alembic_root_path: str,
    engine: Engine,
    base: DeclarativeMeta,
    session: Session,
    alembic_ini_path: Optional[str] = None,
    script_location: Optional[str] = None,
) -> AlembicConfig:
    """Create alembic config.

    Args:
        alembic_root_path: alembic root path
        engine: sqlalchemy engine
        base: sqlalchemy base
        session: sqlalchemy session
        alembic_ini_path (Optional[str]): alembic ini path
        script_location (Optional[str]): alembic script location

    Returns:
        alembic config
    """
    alembic_ini_path = alembic_ini_path or os.path.join(
        alembic_root_path, "alembic.ini"
    )  # 如果未提供 alembic_ini_path，则使用默认的 alembic 根目录下的 alembic.ini 文件路径
    alembic_cfg = AlembicConfig(alembic_ini_path)  # 创建 Alembic 配置对象
    alembic_cfg.set_main_option("sqlalchemy.url", str(engine.url))  # 设置 SQLAlchemy 连接 URL
    script_location = script_location or os.path.join(alembic_root_path, "alembic")
    # 如果未提供 script_location，则使用默认的 alembic 根目录下的 alembic 目录路径
    versions_dir = os.path.join(script_location, "versions")
    
    os.makedirs(script_location, exist_ok=True)  # 创建 alembic 目录，如果目录不存在则创建
    os.makedirs(versions_dir, exist_ok=True)  # 创建版本目录，如果目录不存在则创建
    
    alembic_cfg.set_main_option("script_location", script_location)  # 设置 alembic 配置的脚本路径
    
    alembic_cfg.attributes["target_metadata"] = base.metadata  # 设置 alembic 配置的目标元数据
    alembic_cfg.attributes["session"] = session  # 设置 alembic 配置的会话对象
    return alembic_cfg  # 返回配置对象


def create_migration_script(
    alembic_cfg: AlembicConfig,
    engine: Engine,
    message: str = "New migration",
    create_new_revision_if_noting_to_update: Optional[bool] = True,
) -> str:
    """Create migration script.

    Args:
        alembic_cfg: alembic config
        engine: sqlalchemy engine
        message: migration message
        create_new_revision_if_noting_to_update: Whether to create a new revision if there is nothing to update,
            pass False to avoid creating a new revision if there is nothing to update, default is True

    Returns:
        The path of the generated migration script.
    """
    from alembic.runtime.migration import MigrationContext  # 导入 alembic 运行时迁移上下文
    from alembic.script import ScriptDirectory  # 导入 alembic 脚本目录
    
    # 检查数据库是否是最新状态
    script_dir = ScriptDirectory.from_config(alembic_cfg)  # 根据 alembic 配置创建脚本目录对象
    with engine.connect() as connection:  # 使用引擎连接数据库
        context = MigrationContext.configure(connection=connection)  # 配置迁移上下文
        current_rev = context.get_current_revision()  # 获取当前数据库迁移版本
        head_rev = script_dir.get_current_head()  # 获取脚本目录的当前头部迁移版本
    
    logger.info(
        f"alembic migration current revision: {current_rev}, latest revision: {head_rev}"
    )  # 记录当前数据库迁移版本和脚本目录的最新迁移版本信息
    
    should_create_revision = (
        (current_rev is None and head_rev is None)
        or current_rev != head_rev
        or create_new_revision_if_noting_to_update
    )  # 判断是否需要创建新的迁移版本
    # 如果应该创建新的数据库版本迁移
    if should_create_revision:
        # 使用引擎对象建立数据库连接
        with engine.connect() as connection:
            # 将连接设置到 Alembic 配置对象中的连接属性
            alembic_cfg.attributes["connection"] = connection
            # 使用 Alembic 命令生成新的数据库版本迁移
            revision = command.revision(alembic_cfg, message=message, autogenerate=True)
            # 返回生成的迁移脚本的路径
            return revision.path
    # 如果当前数据库版本等于最新版本
    elif current_rev == head_rev:
        # 记录信息：无需生成迁移脚本，数据库已经是最新的
        logger.info("No migration script to generate, database is up-to-date")
    # 如果没有创建新的版本迁移，返回 None 或者适当的消息
    return None
# 将数据库升级到目标版本的函数
def upgrade_database(
    alembic_cfg: AlembicConfig, engine: Engine, target_version: str = "head"
) -> None:
    """Upgrade database to target version.

    Args:
        alembic_cfg: Alembic configuration object.
        engine: SQLAlchemy engine instance.
        target_version: Target version to upgrade to, default is "head" (latest version).
    """
    # 使用 engine 建立数据库连接
    with engine.connect() as connection:
        # 将连接对象赋给 Alembic 配置对象的 connection 属性
        alembic_cfg.attributes["connection"] = connection
        # 执行升级操作，将数据库升级到目标版本
        command.upgrade(alembic_cfg, target_version)


# 生成升级数据库到目标版本所需的 SQL 脚本的函数
def generate_sql_for_upgrade(
    alembic_cfg: AlembicConfig,
    engine: Engine,
    target_version: Optional[str] = "head",
    output_file: Optional[str] = "migration.sql",
) -> None:
    """Generate SQL for upgrading database to target version.

    Args:
        alembic_cfg: Alembic configuration object.
        engine: SQLAlchemy engine instance.
        target_version: Target version to generate SQL for, default is "head" (latest version).
        output_file: File to write the generated SQL script, default is "migration.sql".
    """
    import contextlib
    import io

    with engine.connect() as connection, contextlib.redirect_stdout(
        io.StringIO()
    ) as stdout:
        # 将连接对象赋给 Alembic 配置对象的 connection 属性
        alembic_cfg.attributes["connection"] = connection
        # 生成 SQL 脚本而不应用更改
        command.upgrade(alembic_cfg, target_version, sql=True)

        # 将生成的 SQL 写入文件
        with open(output_file, "w", encoding="utf-8") as file:
            file.write(stdout.getvalue())


# 降级数据库一个修订版本的函数
def downgrade_database(
    alembic_cfg: AlembicConfig, engine: Engine, revision: str = "-1"
):
    """Downgrade the database by one revision.

    Args:
        alembic_cfg: Alembic configuration object.
        engine: SQLAlchemy engine instance.
        revision: Revision identifier to downgrade to, default is "-1" (one revision back).
    """
    # 使用 engine 建立数据库连接
    with engine.connect() as connection:
        # 将连接对象赋给 Alembic 配置对象的 connection 属性
        alembic_cfg.attributes["connection"] = connection
        # 执行数据库降级操作
        command.downgrade(alembic_cfg, revision)


# 清理 Alembic 迁移脚本和历史记录的函数
def clean_alembic_migration(alembic_cfg: AlembicConfig, engine: Engine) -> None:
    """Clean Alembic migration scripts and history.

    Args:
        alembic_cfg: Alembic configuration object.
        engine: SQLAlchemy engine instance.
    """
    import shutil

    # 获取迁移脚本的存储位置
    script_location = alembic_cfg.get_main_option("script_location")
    print(f"Delete migration script location: {script_location}")

    # 删除所有迁移脚本文件
    for file in os.listdir(script_location):
        if file.startswith("versions"):
            filepath = os.path.join(script_location, file)
            print(f"Delete migration script file: {filepath}")
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath, ignore_errors=True)

    # 如果存在，删除 Alembic 的版本表
    version_table = alembic_cfg.get_main_option("version_table") or "alembic_version"
    # 如果 version_table 参数不为空（即存在版本表名）
    if version_table:
        # 使用 engine 建立数据库连接，并创建连接对象
        with engine.connect() as connection:
            # 打印消息，指示正在删除指定的 Alembic 版本表
            print(f"Delete Alembic version table: {version_table}")
            # 使用 SQL 语句删除指定的表，如果存在的话
            connection.execute(text(f"DROP TABLE IF EXISTS {version_table}"))
    
    # 打印消息，指示已清理 Alembic 迁移脚本和历史记录
    print("Cleaned Alembic migration scripts and history")
_MIGRATION_SOLUTION = """
**Solution 1:**

Run the following command to upgrade the database.

dbgpt db migration upgrade


**Solution 2:**

Run the following command to clean the migration script and migration history.

dbgpt db migration clean -y


**Solution 3:**

If you have already run the above command, but the error still exists, 
you can try the following command to clean the migration script, migration history and your data.
warning: This command will delete all your data!!! Please use it with caution.


dbgpt db migration clean --drop_all_tables -y --confirm_drop_all_tables

or 

rm -rf pilot/meta_data/alembic/versions/*
rm -rf pilot/meta_data/alembic/dbgpt.db


If your database is a shared database, and you run DB-GPT in multiple instances, 
you should make sure that all migration scripts are same in all instances, in this case,
wo strongly recommend you close migration feature by setting `--disable_alembic_upgrade`.
and use `dbgpt db migration` command to manage migration scripts.
"""


def _check_database_migration_status(alembic_cfg: AlembicConfig, engine: Engine):
    """Check if the database is at the latest migration revision.

    If your database is a shared database, and you run DB-GPT in multiple instances,
    you should make sure that all migration scripts are same in all instances, in this case,
    wo strongly recommend you close migration feature by setting `disable_alembic_upgrade` to True.
    and use `dbgpt db migration` command to manage migration scripts.

    Args:
        alembic_cfg: Alembic configuration object.
        engine: SQLAlchemy engine instance.
    Raises:
        Exception: If the database is not at the latest revision.
    """
    # Import necessary modules from Alembic for managing database migrations
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory

    # Create a ScriptDirectory object based on the provided Alembic configuration
    script = ScriptDirectory.from_config(alembic_cfg)

    # Define a function to retrieve the current migration revision of the database
    def get_current_revision(engine):
        with engine.connect() as connection:
            context = MigrationContext.configure(connection=connection)
            return context.get_current_revision()

    # Get the current revision of the database and the latest head revision from the migration scripts
    current_rev = get_current_revision(engine)
    head_rev = script.get_current_head()

    # Prepare informational message about migration versions and their file paths
    script_info_msg = "Migration versions and their file paths:"
    script_info_msg += f"\n{'='*40}Migration versions{'='*40}\n"

    # Iterate through all migration revisions to gather details
    for revision in script.walk_revisions(base="base"):
        current_marker = "(current)" if revision.revision == current_rev else ""
        script_path = script.get_revision(revision.revision).path
        script_info_msg += f"\n{revision.revision} {current_marker}: {revision.doc} (Path: {script_path})"

    script_info_msg += f"\n{'='*90}"

    # Log the gathered migration information
    logger.info(script_info_msg)
    # 如果当前数据库的修订版本号不等于最新的修订版本号，则执行以下操作
    logger.error(
        "Database is not at the latest revision. "
        f"Current revision: {current_rev}, latest revision: {head_rev}\n"
        "Please apply existing migration scripts before generating new ones. "
        "Check the listed file paths for migration scripts.\n"
        f"Also you can try the following solutions:\n{_MIGRATION_SOLUTION}\n"
    )
    # 抛出异常，指示数据库迁移状态检查失败，并提供错误信息和解决方案建议
    raise Exception(
        "Check database migration status failed, you can see the error and solutions above"
    )
def _get_latest_revision(alembic_cfg: AlembicConfig, engine: Engine) -> str:
    """Get the latest revision of the database.

    Args:
        alembic_cfg: Alembic configuration object.
        engine: SQLAlchemy engine instance.

    Returns:
        The latest revision as a string.
    """
    # Import the MigrationContext class from alembic runtime
    from alembic.runtime.migration import MigrationContext

    # Establish a connection to the database engine
    with engine.connect() as connection:
        # Configure MigrationContext with the established connection
        context = MigrationContext.configure(connection=connection)
        # Retrieve and return the current revision of the database
        return context.get_current_revision()


def _delete_migration_script(script_path: str):
    """Delete a migration script.

    Args:
        script_path: The path of the migration script to delete.
    """
    # Check if the specified script path exists
    if os.path.exists(script_path):
        # Delete the script file from the filesystem
        os.remove(script_path)
        # Log deletion information
        logger.info(f"Deleted migration script at: {script_path}")
    else:
        # Log a warning if the script path does not exist
        logger.warning(f"Migration script not found at: {script_path}")


def _ddl_init_and_upgrade(
    default_meta_data_path: str,
    disable_alembic_upgrade: bool,
    alembic_ini_path: Optional[str] = None,
    script_location: Optional[str] = None,
):
    """Initialize and upgrade database metadata

    Args:
        default_meta_data_path (str): default meta data path
        disable_alembic_upgrade (bool): Whether to enable alembic to initialize and upgrade database metadata
        alembic_ini_path (Optional[str]): alembic ini path
        script_location (Optional[str]): alembic script location
    """
    # Check if alembic upgrade is disabled
    if disable_alembic_upgrade:
        # Log a message indicating alembic upgrade is disabled and return
        logger.info(
            "disable_alembic_upgrade is true, not to initialize and upgrade database metadata with alembic"
        )
        return
    else:
        # Warn about initializing and upgrading database metadata with alembic in development environment
        warn_msg = (
            "Initialize and upgrade database metadata with alembic, "
            "just run this in your development environment, if you deploy this in production environment, "
            "please run webserver with --disable_alembic_upgrade(`python dbgpt/app/dbgpt_server.py "
            "--disable_alembic_upgrade`).\n"
            "we suggest you to use `dbgpt db migration` to initialize and upgrade database metadata with alembic, "
            "your can run `dbgpt db migration --help` to get more information."
        )
        logger.warning(warn_msg)

    # Import necessary modules from dbgpt storage metadata db_manager
    from dbgpt.storage.metadata.db_manager import db

    # Create alembic configuration using utility function
    alembic_cfg = create_alembic_config(
        default_meta_data_path,
        db.engine,
        db.Model,
        db.session(),
        alembic_ini_path,
        script_location,
    )

    try:
        # Check the current status of database migration
        _check_database_migration_status(alembic_cfg, db.engine)
    except Exception as e:
        # Log error if checking migration status fails and raise the exception
        logger.error(f"Failed to check database migration status: {e}")
        raise

    # Placeholder for the latest revision before any changes
    latest_revision_before = "__latest_revision_before__"
    # Placeholder for new script path, initially set to None
    new_script_path = None
    try:
        # 获取当前数据库的最新修订版本号
        latest_revision_before = _get_latest_revision(alembic_cfg, db.engine)
        # 创建迁移脚本，如果没有需要更新的内容，则设置 create_new_revision_if_noting_to_update=False，避免创建大量空的迁移脚本
        # TODO 设置 create_new_revision_if_noting_to_update=False，目前未生效。
        new_script_path = create_migration_script(
            alembic_cfg, db.engine, create_new_revision_if_noting_to_update=True
        )
        # 使用 alembic 进行数据库升级
        upgrade_database(alembic_cfg, db.engine)
    except CommandError as e:
        # 如果错误信息指示目标数据库不是最新的，则记录错误日志并提供可能的解决方案
        if "Target database is not up to date" in str(e):
            logger.error(
                f"Initialize and upgrade database metadata with alembic failed, error detail: {str(e)} "
                f"you can try the following solutions:\n{_MIGRATION_SOLUTION}\n"
            )
            # 抛出异常，说明初始化和升级数据库元数据失败，提供错误信息和解决方案
            raise Exception(
                "Initialize and upgrade database metadata with alembic failed, "
                "you can see the error and solutions above"
            ) from e
        else:
            # 获取升级后的最新数据库修订版本号
            latest_revision_after = _get_latest_revision(alembic_cfg, db.engine)
            # 如果升级前后的最新修订版本号不一致，则记录错误日志
            if latest_revision_before != latest_revision_after:
                logger.error(
                    f"Upgrade database failed. Please review the migration script manually. "
                    f"Failed script path: {new_script_path}\nError: {e}"
                )
            # 抛出原始的 CommandError 异常
            raise e
```