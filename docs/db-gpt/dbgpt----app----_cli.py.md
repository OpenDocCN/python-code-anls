# `.\DB-GPT-src\dbgpt\app\_cli.py`

```py
import functools  # 导入 functools 模块，用于高阶函数的工具函数
import os  # 导入 os 模块，提供与操作系统交互的功能
from typing import Optional  # 导入 Optional 类型提示，用于声明可选类型的参数

import click  # 导入 click 库，用于创建命令行接口

from dbgpt.app.base import WebServerParameters  # 导入 WebServerParameters 类，定义了 Web 服务器的参数
from dbgpt.configs.model_config import LOGDIR  # 从配置中导入 LOGDIR 变量，用于日志目录的路径
from dbgpt.util.command_utils import _run_current_with_daemon, _stop_service  # 导入命令行实用程序中的两个函数
from dbgpt.util.parameter_utils import EnvArgumentParser  # 导入环境参数解析器

@click.command(name="webserver")  # 创建名为 "webserver" 的命令行命令
@EnvArgumentParser.create_click_option(WebServerParameters)  # 使用 EnvArgumentParser 创建与 WebServerParameters 相关的命令行选项
def start_webserver(**kwargs):
    """Start webserver(dbgpt_server.py)"""
    if kwargs["daemon"]:
        log_file = os.path.join(LOGDIR, "webserver_uvicorn.log")  # 根据 LOGDIR 和文件名创建日志文件路径
        _run_current_with_daemon("WebServer", log_file)  # 使用守护进程模式运行 WebServer
    else:
        from dbgpt.app.dbgpt_server import run_webserver  # 导入运行 Web 服务器的函数

        run_webserver(WebServerParameters(**kwargs))  # 运行 Web 服务器并传入参数

@click.command(name="webserver")  # 创建名为 "webserver" 的命令行命令
@click.option(
    "--port",
    type=int,
    default=None,
    required=False,
    help=("The port to stop"),
)
def stop_webserver(port: int):
    """Stop webserver(dbgpt_server.py)"""
    _stop_service("webserver", "WebServer", port=port)  # 停止 Web 服务器服务，传入端口参数

def _stop_all_dbgpt_server():
    _stop_service("webserver", "WebServer")  # 停止所有 dbgpt 服务器服务

@click.group("migration")  # 创建名为 "migration" 的命令组
def migration():
    """Manage database migration"""
    pass  # 空操作，用于管理数据库迁移命令

def add_migration_options(func):
    @click.option(
        "--alembic_ini_path",
        required=False,
        type=str,
        default=None,
        show_default=True,
        help="Alembic ini path, if not set, use 'pilot/meta_data/alembic.ini'",
    )
    @click.option(
        "--script_location",
        required=False,
        type=str,
        default=None,
        show_default=True,
        help="Alembic script location, if not set, use 'pilot/meta_data/alembic'",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper  # 返回装饰器函数

@migration.command()  # 将装饰器应用到 migration 命令组中的命令
@add_migration_options  # 使用 add_migration_options 装饰器为命令添加选项
@click.option(
    "-m",
    "--message",
    required=False,
    type=str,
    default="Init migration",
    show_default=True,
    help="The message for create migration repository",
)
def init(alembic_ini_path: str, script_location: str, message: str):
    """Initialize database migration repository"""
    from dbgpt.util._db_migration_utils import create_migration_script  # 导入创建数据库迁移脚本的函数

    alembic_cfg, db_manager = _get_migration_config(alembic_ini_path, script_location)  # 获取迁移配置和数据库管理器
    create_migration_script(alembic_cfg, db_manager.engine, message)  # 创建迁移脚本，传入配置、引擎和消息

@migration.command()  # 将装饰器应用到 migration 命令组中的命令
@add_migration_options  # 使用 add_migration_options 装饰器为命令添加选项
@click.option(
    "-m",
    "--message",
    required=False,
    type=str,
    default="New migration",
    show_default=True,
    help="The message for migration script",
)
def migrate(alembic_ini_path: str, script_location: str, message: str):
    """Create migration script"""
    from dbgpt.util._db_migration_utils import create_migration_script  # 导入创建数据库迁移脚本的函数

    alembic_cfg, db_manager = _get_migration_config(alembic_ini_path, script_location)  # 获取迁移配置和数据库管理器
    create_migration_script(alembic_cfg, db_manager.engine, message)  # 创建迁移脚本，传入配置、引擎和消息

@migration.command()  # 将装饰器应用到 migration 命令组中的命令
@add_migration_options  # 使用 add_migration_options 装饰器为命令添加选项
@click.option(
    "--sql-output",
    # 定义命令行参数的类型为字符串
    type=str,
    # 设置默认值为None，表示参数是可选的
    default=None,
    # 提供帮助信息，指示该参数用于生成迁移的SQL脚本而不是应用迁移
    help="Generate SQL script for migration instead of applying it. ex: --sql-output=upgrade.sql",
# 定义数据库升级函数，接受 Alembic 配置文件路径、脚本位置和 SQL 输出路径作为参数
def upgrade(alembic_ini_path: str, script_location: str, sql_output: str):
    """Upgrade database to target version"""
    # 导入数据库升级相关的工具函数
    from dbgpt.util._db_migration_utils import (
        generate_sql_for_upgrade,
        upgrade_database,
    )

    # 获取 Alembic 配置和数据库管理器对象
    alembic_cfg, db_manager = _get_migration_config(alembic_ini_path, script_location)
    
    # 如果指定了 SQL 输出路径，则生成升级 SQL 文件
    if sql_output:
        generate_sql_for_upgrade(alembic_cfg, db_manager.engine, output_file=sql_output)
    else:
        # 否则直接升级数据库
        upgrade_database(alembic_cfg, db_manager.engine)


# 定义数据库降级命令函数，接受 Alembic 配置文件路径、脚本位置、确认标志和降级版本作为参数
@migration.command()
@add_migration_options
@click.option(
    "-y",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Confirm to downgrade database",
)
@click.option(
    "-r",
    "--revision",
    default="-1",
    show_default=True,
    help="Revision to downgrade to",
)
def downgrade(alembic_ini_path: str, script_location: str, y: bool, revision: str):
    """Downgrade database to target version"""
    # 导入数据库降级相关的工具函数
    from dbgpt.util._db_migration_utils import downgrade_database

    # 如果没有确认降级操作，则提示用户确认
    if not y:
        click.confirm("Are you sure you want to downgrade the database?", abort=True)
    
    # 获取 Alembic 配置和数据库管理器对象
    alembic_cfg, db_manager = _get_migration_config(alembic_ini_path, script_location)
    
    # 执行数据库降级操作
    downgrade_database(alembic_cfg, db_manager.engine, revision)


# 定义清理 Alembic 迁移脚本和历史记录命令函数，接受 Alembic 配置文件路径、脚本位置、清理选项和确认标志作为参数
@migration.command()
@add_migration_options
@click.option(
    "--drop_all_tables",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Drop all tables",
)
@click.option(
    "-y",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Confirm to clean migration data",
)
@click.option(
    "--confirm_drop_all_tables",
    required=False,
    type=bool,
    default=False,
    is_flag=True,
    help="Confirm to drop all tables",
)
def clean(
    alembic_ini_path: str,
    script_location: str,
    drop_all_tables: bool,
    y: bool,
    confirm_drop_all_tables: bool,
):
    """Clean Alembic migration scripts and history"""
    # 导入清理 Alembic 迁移相关的工具函数
    from dbgpt.util._db_migration_utils import clean_alembic_migration

    # 如果没有确认清理操作，则提示用户确认
    if not y:
        click.confirm(
            "Are you sure clean alembic migration scripts and history?", abort=True
        )
    
    # 获取 Alembic 配置和数据库管理器对象
    alembic_cfg, db_manager = _get_migration_config(alembic_ini_path, script_location)
    
    # 执行清理 Alembic 迁移的操作
    clean_alembic_migration(alembic_cfg, db_manager.engine)
    
    # 如果选择了删除所有表的选项，并且没有确认删除操作，则提示用户确认
    if drop_all_tables:
        if not confirm_drop_all_tables:
            click.confirm("\nAre you sure drop all tables?", abort=True)
        
        # 使用数据库连接删除所有表格
        with db_manager.engine.connect() as connection:
            for tbl in reversed(db_manager.Model.metadata.sorted_tables):
                print(f"Drop table {tbl.name}")
                connection.execute(tbl.delete())


# 定义列出所有迁移版本命令函数，接受 Alembic 配置文件路径和脚本位置作为参数
@migration.command()
@add_migration_options
def list(alembic_ini_path: str, script_location: str):
    """List all versions in the migration history, marking the current one"""
    # 导入 Alembic 迁移运行时和脚本目录相关的模块
    from alembic.runtime.migration import MigrationContext
    from alembic.script import ScriptDirectory
    # 调用函数获取 Alembic 配置对象和数据库管理器
    alembic_cfg, db_manager = _get_migration_config(alembic_ini_path, script_location)

    # 使用 Alembic 配置对象创建脚本目录对象
    script = ScriptDirectory.from_config(alembic_cfg)

    # 定义函数：获取当前数据库的迁移版本
    def get_current_revision():
        # 使用数据库管理器的引擎对象建立连接
        with db_manager.engine.connect() as connection:
            # 配置迁移上下文
            context = MigrationContext.configure(connection)
            # 返回当前数据库的迁移版本
            return context.get_current_revision()

    # 调用函数获取当前数据库的迁移版本
    current_rev = get_current_revision()

    # 遍历脚本目录中的所有迁移版本，并标记当前版本
    for revision in script.walk_revisions():
        # 如果迁移版本与当前版本相同，则添加标记 "(current)"
        current_marker = "(current)" if revision.revision == current_rev else ""
        # 打印迁移版本号、标记和文档说明
        print(f"{revision.revision} {current_marker}: {revision.doc}")
# 定义一个命令函数 `show`，用于显示特定版本的迁移脚本。
@migration.command()
@add_migration_options
@click.argument("revision", required=True)
def show(alembic_ini_path: str, script_location: str, revision: str):
    """Show the migration script for a specific version."""
    # 导入 Alembic 的 ScriptDirectory 类
    from alembic.script import ScriptDirectory

    # 获取迁移配置和数据库管理器对象
    alembic_cfg, db_manager = _get_migration_config(alembic_ini_path, script_location)

    # 使用配置创建 ScriptDirectory 对象
    script = ScriptDirectory.from_config(alembic_cfg)

    # 获取指定版本的迁移脚本对象
    rev = script.get_revision(revision)
    if rev is None:
        # 如果找不到指定版本的迁移脚本，则打印提示信息并返回
        print(f"Revision {revision} not found.")
        return

    # 查找包含迁移脚本文件的目录
    script_files = os.listdir(os.path.join(script.dir, "versions"))
    # 找到以指定版本号开头的迁移脚本文件
    script_file = next((f for f in script_files if f.startswith(revision)), None)

    if script_file is None:
        # 如果找不到指定版本号开头的迁移脚本文件，则打印提示信息并返回
        print(f"Migration script for revision {revision} not found.")
        return

    # 构建完整的迁移脚本文件路径
    script_file_path = os.path.join(script.dir, "versions", script_file)
    # 打印指定版本号的迁移脚本文件路径
    print(f"Migration script for revision {revision}: {script_file_path}")

    try:
        # 尝试打开并读取迁移脚本文件内容
        with open(script_file_path, "r") as file:
            print(file.read())
    except FileNotFoundError:
        # 如果迁移脚本文件不存在，则打印文件未找到的错误信息
        print(f"Migration script {script_file_path} not found.")


# 定义一个内部函数 `_get_migration_config`，用于获取迁移配置
def _get_migration_config(
    alembic_ini_path: Optional[str] = None, script_location: Optional[str] = None
):
    # 导入 `_initialize_db` 函数，用于数据库初始化
    from dbgpt.app.base import _initialize_db

    # 导入所有模型，确保它们在 SQLAlchemy 中注册
    from dbgpt.app.initialization.db_model_initialization import _MODELS
    # 导入数据库管理器对象
    from dbgpt.storage.metadata.db_manager import db as db_manager
    # 导入创建 Alembic 配置的函数
    from dbgpt.util._db_migration_utils import create_alembic_config

    # 初始化数据库，获取默认的元数据路径
    default_meta_data_path = _initialize_db()

    # 创建 Alembic 配置对象
    alembic_cfg = create_alembic_config(
        default_meta_data_path,
        db_manager.engine,
        db_manager.Model,
        db_manager.session(),
        alembic_ini_path,
        script_location,
    )

    # 返回 Alembic 配置对象和数据库管理器对象
    return alembic_cfg, db_manager
```