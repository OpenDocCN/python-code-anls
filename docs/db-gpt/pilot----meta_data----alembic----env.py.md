# `.\DB-GPT-src\pilot\meta_data\alembic\env.py`

```py
# 从 alembic 模块中导入 context 对象，用于管理 Alembic 迁移
from alembic import context
# 从 sqlalchemy 模块中导入 engine_from_config 和 pool
from sqlalchemy import engine_from_config, pool
# 从 dbgpt.storage.metadata.db_manager 模块中导入 db 对象
from dbgpt.storage.metadata.db_manager import db

# 获取当前 Alembic 配置对象，用于访问使用的 .ini 文件中的值
config = context.config

# 下面的部分用于添加你的模型的 MetaData 对象，以支持 'autogenerate'
# 例如：from myapp import mymodel

# 从配置中获取其它需要在 env.py 中使用的值
# 例如：my_important_option = config.get_main_option("my_important_option")
# ... 等等


def run_migrations_offline() -> None:
    """以 'offline' 模式运行迁移。

    这将使用仅 URL 而不是 Engine 配置上下文，尽管在这里也可以接受 Engine。
    通过跳过 Engine 的创建，我们甚至不需要一个可用的 DBAPI。

    在此处调用 context.execute() 将把给定的字符串输出到脚本输出。

    """
    # 获取目标元数据对象，通常从 db 对象中获取
    target_metadata = db.metadata
    # 从配置中获取 SQLAlchemy 连接 URL
    url = config.get_main_option("sqlalchemy.url")
    # 断言目标元数据和 URL 都不为 None
    assert target_metadata is not None
    assert url is not None
    # 配置 Alembic 上下文
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    # 使用事务运行迁移脚本
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """以 'online' 模式运行迁移。

    在这种情况下，我们需要创建一个 Engine，并将连接与上下文关联起来。

    """
    # 获取数据库引擎对象
    engine = db.engine
    # 获取目标元数据对象
    target_metadata = db.metadata
    # 使用 Engine 连接数据库
    with engine.connect() as connection:
        # 根据 Engine 的方言配置 Alembic 上下文
        if engine.dialect.name == "sqlite":
            context.configure(
                connection=engine.connect(),  # 为 SQLite 使用独立的连接
                target_metadata=target_metadata,
                render_as_batch=True,  # 批处理模式
            )
        else:
            context.configure(connection=connection, target_metadata=target_metadata)

        # 使用事务运行迁移脚本
        with context.begin_transaction():
            context.run_migrations()


# 根据当前是否处于离线模式选择运行相应的迁移函数
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```