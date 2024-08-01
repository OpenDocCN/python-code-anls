# `.\DB-GPT-src\dbgpt\storage\metadata\db_manager.py`

```py
# 导入必要的模块和库
from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import ClassVar, Dict, Generic, Iterator, Optional, Type, TypeVar, Union
# 导入 SQLAlchemy 相关模块
from sqlalchemy import URL, Engine, MetaData, create_engine, inspect, orm
from sqlalchemy.orm import (
    DeclarativeMeta,
    Session,
    declarative_base,
    scoped_session,
    sessionmaker,
)
from sqlalchemy.pool import QueuePool
# 导入自定义工具类
from dbgpt.util.pagination_utils import PaginationResult
from dbgpt.util.string_utils import _to_str

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)
# 定义类型变量 T
T = TypeVar("T", bound="BaseModel")

# 基础查询类，继承自 SQLAlchemy 的 Query 类
class BaseQuery(orm.Query):
    """Base query class."""

    # 分页查询方法
    def paginate_query(self, page: int = 1, per_page: int = 20) -> PaginationResult:
        """Paginate the query.

        Example:

            .. code-block:: python

                from dbgpt.storage.metadata import db, Model


                class User(Model):
                    __tablename__ = "user"
                    id = Column(Integer, primary_key=True)
                    name = Column(String(50))
                    fullname = Column(String(50))


                with db.session() as session:
                    pagination = session.query(User).paginate_query(
                        page=1, page_size=10
                    )
                    print(pagination)

        Args:
            page (Optional[int], optional): The page number. Defaults to 1.
            per_page (Optional[int], optional): The number of items per page. Defaults
                to 20.
        Returns:
            PaginationResult: The pagination result.
        """
        # 检查页码和每页数量是否合法
        if page < 1:
            raise ValueError("Page must be greater than 0")
        if per_page < 0:
            raise ValueError("Per page must be greater than 0")
        # 执行分页查询
        items = self.limit(per_page).offset((page - 1) * per_page).all()
        total = self.order_by(None).count()
        total_pages = (total - 1) // per_page + 1
        # 返回分页结果对象
        return PaginationResult(
            items=items,
            total_count=total,
            total_pages=total_pages,
            page=page,
            page_size=per_page,
        )

# 基础模型类
class _Model:
    """Base class for SQLAlchemy declarative base model."""

    # 数据库管理器
    __db_manager__: ClassVar[DatabaseManager]
    # 查询类
    query_class = BaseQuery

    # 对象字符串表示方法
    def __repr__(self):
        identity = inspect(self).identity
        if identity is None:
            pk = "(transient {0})".format(id(self))
        else:
            pk = ", ".join(_to_str(value) for value in identity)
        return "<{0} {1}>".format(type(self).__name__, pk)

# 数据库管理器类
class DatabaseManager:
    """The database manager.
    # 定义一个基础查询类，用于执行数据库查询操作
    Query = BaseQuery

    def __init__(self):
        """Create a DatabaseManager."""
        # 初始化 DatabaseManager 类
        self._db_url = None  # 数据库连接 URL，默认为空
        # 创建一个声明性基类，用于定义数据库模型
        self._base: DeclarativeMeta = self._make_declarative_base(_Model)
        self._engine: Optional[Engine] = None  # 引擎对象，默认为空
        self._session: Optional[scoped_session] = None  # 数据库会话对象，默认为空

    @property
    def Model(self) -> _Model:
        """Get the declarative base."""
        return self._base  # 返回声明性基类对象，用于定义数据库模型

    @property
    def metadata(self) -> MetaData:
        """Get the metadata."""
        return self.Model.metadata  # 返回模型的元数据对象

    @property
    def engine(self):
        """Get the engine.""" ""
        return self._engine  # 返回数据库引擎对象

    @property
    def is_initialized(self) -> bool:
        """Whether the database manager is initialized."""
        return self._engine is not None and self._session is not None
        # 返回数据库管理器是否已经初始化的布尔值

    @contextmanager
        """获取具有上下文管理器的会话对象。

        此上下文管理器处理 SQLAlchemy 会话的生命周期。根据执行情况自动提交或回滚事务，并处理会话关闭。

        参数 `commit` 控制会话是否应在块结束时提交更改。这对于分离读取和写入操作很有用。

        示例：
            .. code-block:: python

                # 对于写入操作（插入、更新、删除）：
                with db.session() as session:
                    user = User(name="John Doe")
                    session.add(user)
                    # session.commit() 将自动调用

                # 对于只读操作：
                with db.session(commit=False) as session:
                    user = session.query(User).filter_by(name="John Doe").first()
                    # 由于只读操作，不需要调用 session.commit()

        Args:
            commit (Optional[bool], optional): 是否提交会话。如果为 True（默认），会话将在块结束时提交更改。
                对于只读操作或需要手动控制提交的情况，请使用 False。默认为 True。

        Yields:
            Session: SQLAlchemy 会话对象。

        Raises:
            RuntimeError: 如果数据库管理器未初始化，则引发此异常。
            Exception: 传播块内发生的任何异常。

        重要提示：
            - DetachedInstanceError: 当尝试访问或修改已从其会话中分离的实例时会发生此错误。
              当会话关闭并尝试访问延迟加载属性时，特别是在尝试进一步与 ORM 对象交互时，可能会发生 DetachedInstanceError。
              为避免此错误：
                a. 确保在会话关闭之前加载所需的属性。
                b. 避免在所有必要的 ORM 对象交互完成之前关闭会话。
                c. 如果会话关闭后需要进一步与实例交互，请将实例重新绑定到新会话。
        """
        # 如果数据库管理器未初始化，则抛出 RuntimeError 异常
        if not self.is_initialized:
            raise RuntimeError("The database manager is not initialized.")
        # 调用私有方法 _session() 获取会话对象
        session = self._session()  # type: ignore
        try:
            # 使用 yield 返回会话对象，进入上下文管理器的主体
            yield session
            # 如果 commit 参数为 True，则提交会话中的事务更改
            if commit:
                session.commit()
        except Exception:
            # 如果发生异常，则回滚会话中的事务更改，并重新抛出异常
            session.rollback()
            raise
        finally:
            # 最终关闭会话，确保资源释放
            session.close()
    ) -> DeclarativeMeta:
        """Make the declarative base.

        Args:
            model (DeclarativeMeta): The base class.

        Returns:
            DeclarativeMeta: The declarative base.
        """
        # 如果传入的 model 不是 DeclarativeMeta 类型，则使用 declarative_base 函数创建一个默认的 DeclarativeMeta 类型
        if not isinstance(model, DeclarativeMeta):
            model = declarative_base(cls=model, name="Model")
        # 如果 model 没有定义 query_class 属性，则将 self.Query 赋值给 model.query_class 属性
        if not getattr(model, "query_class", None):
            model.query_class = self.Query  # type: ignore
        # 将当前对象 self 赋值给 model.__db_manager__ 属性
        model.__db_manager__ = self  # type: ignore
        # 返回创建好的 model 对象作为结果
        return model  # type: ignore

    def init_db(
        self,
        db_url: Union[str, URL],
        engine_args: Optional[Dict] = None,
        base: Optional[DeclarativeMeta] = None,
        query_class=BaseQuery,
        override_query_class: Optional[bool] = False,
        session_options: Optional[Dict] = None,
    ):
        """Initialize the database manager.

        Args:
            db_url (Union[str, URL]): The database url.
            engine_args (Optional[Dict], optional): The engine arguments. Defaults to
                None.
            base (Optional[DeclarativeMeta]): The base class. Defaults to None.
            query_class (BaseQuery, optional): The query class. Defaults to BaseQuery.
            override_query_class (Optional[bool], optional): Whether to override the
                query class. Defaults to False.
            session_options (Optional[Dict], optional): The session options. Defaults
                to None.
        """
        # 如果 session_options 为 None，则将其设为空字典
        if session_options is None:
            session_options = {}
        # 将传入的 db_url 赋值给 self._db_url 属性
        self._db_url = db_url
        # 如果传入了 query_class，则将其赋值给 self.Query
        if query_class is not None:
            self.Query = query_class
        # 如果传入了 base 类，则设置 self._base 为传入的 base 类，并根据需要设置 base 的 query_class 和 __db_manager__ 属性
        if base is not None:
            self._base = base
            # 如果 base 没有定义 query_class 属性或需要覆盖，则将 self.Query 赋值给 base.query_class
            if not getattr(base, "query_class", None) or override_query_class:
                base.query_class = self.Query
            # 如果 base 没有定义 __db_manager__ 属性或需要覆盖，则将 self 赋值给 base.__db_manager__
            if not hasattr(base, "__db_manager__") or override_query_class:
                base.__db_manager__ = self
        # 根据 db_url 和 engine_args 创建数据库引擎对象并赋值给 self._engine
        self._engine = create_engine(db_url, **(engine_args or {}))

        # 设置默认的会话类和查询类，并创建 session_factory
        session_options.setdefault("class_", Session)
        session_options.setdefault("query_cls", self.Query)
        session_factory = sessionmaker(bind=self._engine, **session_options)
        # 将 session_factory 赋值给 self._session
        self._session = session_factory  # type: ignore
        # 将数据库引擎对象绑定到 self._base.metadata
        self._base.metadata.bind = self._engine  # type: ignore

    def init_default_db(
        self,
        sqlite_path: str,
        engine_args: Optional[Dict] = None,
        base: Optional[DeclarativeMeta] = None,
    ):
        """
        Initialize the database manager with default configuration.

        Examples:
            >>> db.init_default_db(sqlite_path)
            >>> with db.session() as session:
            ...     session.query(...)
            ...

        Args:
            sqlite_path (str): The sqlite path.
            engine_args (Optional[Dict], optional): The engine arguments.
                Defaults to None, if None, we will use connection pool.
            base (Optional[DeclarativeMeta]): The base class. Defaults to None.
        """
        if not engine_args:
            engine_args = {
                # Pool class
                "poolclass": QueuePool,
                # The number of connections to keep open inside the connection pool.
                "pool_size": 10,
                # The maximum overflow size of the pool when the number of connections
                # in the pool is exceeded (pool_size).
                "max_overflow": 20,
                # The number of seconds to wait before giving up on getting a connection
                # from the pool.
                "pool_timeout": 30,
                # Recycle the connection if it has been idle for this many seconds.
                "pool_recycle": 3600,
                # Enable the connection pool “pre-ping” feature that tests connections
                # for liveness upon each checkout.
                "pool_pre_ping": True,
            }

        self.init_db(f"sqlite:///{sqlite_path}", engine_args, base)

    def create_all(self):
        """
        Create all tables.
        """
        self.Model.metadata.create_all(self._engine)

    @staticmethod
    def build_from(
        db_url_or_db: Union[str, URL, DatabaseManager],
        engine_args: Optional[Dict] = None,
        base: Optional[DeclarativeMeta] = None,
        query_class=BaseQuery,
        override_query_class: Optional[bool] = False,
    ) -> DatabaseManager:
        """Build the database manager from the db_url_or_db.

        Examples:
            Build from the database url.
            .. code-block:: python

                from dbgpt.storage.metadata import DatabaseManager
                from sqlalchemy import Column, Integer, String

                db = DatabaseManager.build_from("sqlite:///:memory:")

                class User(db.Model):
                    __tablename__ = "user"
                    id = Column(Integer, primary_key=True)
                    name = Column(String(50))
                    fullname = Column(String(50))

                db.create_all()
                with db.session() as session:
                    session.add(User(name="test", fullname="test"))
                    session.commit()
                    print(User.query.filter(User.name == "test").all())

        Args:
            db_url_or_db (Union[str, URL, DatabaseManager]): The database url or the
                database manager.
            engine_args (Optional[Dict], optional): The engine arguments. Defaults to
                None.
            base (Optional[DeclarativeMeta]): The base class. Defaults to None.
            query_class (BaseQuery, optional): The query class. Defaults to BaseQuery.
            override_query_class (Optional[bool], optional): Whether to override the
                query class. Defaults to False.

        Returns:
            DatabaseManager: The database manager.

        Raises:
            ValueError: If db_url_or_db is neither a string nor a DatabaseManager.

        """
        # 如果 db_url_or_db 是字符串或 URL 对象，则创建一个新的 DatabaseManager 实例
        if isinstance(db_url_or_db, (str, URL)):
            db_manager = DatabaseManager()
            # 使用提供的 URL 初始化数据库连接
            db_manager.init_db(
                db_url_or_db, engine_args, base, query_class, override_query_class
            )
            return db_manager
        # 如果 db_url_or_db 已经是一个 DatabaseManager 实例，则直接返回它
        elif isinstance(db_url_or_db, DatabaseManager):
            return db_url_or_db
        # 如果 db_url_or_db 不是预期的类型，则抛出 ValueError 异常
        else:
            raise ValueError(
                f"db_url_or_db should be either url or a DatabaseManager, got "
                f"{type(db_url_or_db)}"
            )
db = DatabaseManager()
"""The global database manager."""

class BaseCRUDMixin(Generic[T]):
    """The base CRUD mixin."""
    
    __abstract__ = True

    @classmethod
    def db(cls) -> DatabaseManager:
        """Get the database manager."""
        return cls.__db_manager__  # type: ignore


class BaseModel(BaseCRUDMixin[T], _Model, Generic[T]):
    """The base model class that includes CRUD convenience methods."""

    __abstract__ = True
    """Whether the model is abstract."""


def create_model(db_manager: DatabaseManager) -> Type[BaseModel[T]]:
    """Create a model."""
    
    class CRUDMixin(BaseCRUDMixin[T], Generic[T]):
        """Mixin that adds convenience methods for CRUD."""

        _db_manager: DatabaseManager = db_manager

        @classmethod
        def set_db(cls, db_manager: DatabaseManager):
            # TODO: It is hard to replace to user DB Connection
            cls._db_manager = db_manager

        @classmethod
        def db(cls) -> DatabaseManager:
            """Get the database manager."""
            return cls._db_manager

    class _NewModel(CRUDMixin[T], db_manager.Model, Generic[T]):
        """Base model class that includes CRUD convenience methods."""

        __abstract__ = True

    return _NewModel


Model: Type = create_model(db)


def initialize_db(
    db_url: Union[str, URL],
    db_name: str,
    engine_args: Optional[Dict] = None,
    base: Optional[DeclarativeMeta] = None,
    try_to_create_db: Optional[bool] = False,
    session_options: Optional[Dict] = None,
) -> DatabaseManager:
    """Initialize the database manager."""
    def create_database(db_url, db_name, engine_args=None, base=None, try_to_create_db=False, session_options=None):
        """
        Create a database manager and optionally try to create the database schema.
    
        Args:
            db_url (Union[str, URL]): The database url.
            db_name (str): The database name.
            engine_args (Optional[Dict], optional): The engine arguments. Defaults to None.
            base (Optional[DeclarativeMeta]): The base class. Defaults to None.
            try_to_create_db (Optional[bool], optional): Whether to try to create the
                database. Defaults to False.
            session_options (Optional[Dict], optional): The session options. Defaults to
                None.
        Returns:
            DatabaseManager: The database manager.
        """
        # Initialize the database with provided parameters
        db.init_db(db_url, engine_args, base, session_options=session_options)
        
        # If try_to_create_db is True, attempt to create all tables
        if try_to_create_db:
            try:
                db.create_all()
            except Exception as e:
                # Log an error message if database creation fails
                logger.error(f"Failed to create database {db_name}: {e}")
        
        # Return the initialized database manager instance
        return db
```