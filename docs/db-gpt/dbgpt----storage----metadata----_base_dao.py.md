# `.\DB-GPT-src\dbgpt\storage\metadata\_base_dao.py`

```py
from contextlib import contextmanager
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, Union

from sqlalchemy.orm.session import Session

from dbgpt._private.pydantic import model_to_dict
from dbgpt.util.pagination_utils import PaginationResult

from .db_manager import BaseQuery, DatabaseManager, db

# 定义类型变量 T，表示实体类型
T = TypeVar("T")
# 定义类型变量 REQ，表示请求数据结构类型
REQ = TypeVar("REQ")
# 定义类型变量 RES，表示响应数据结构类型
RES = TypeVar("RES")

# 定义查询规格类型 QUERY_SPEC，可以是 REQ 类型或者包含任意键值对的字典
QUERY_SPEC = Union[REQ, Dict[str, Any]]


class BaseDao(Generic[T, REQ, RES]):
    """所有 DAO 的基类。

    Examples:
        .. code-block:: python

            class UserDao(BaseDao):
                def get_user_by_name(self, name: str) -> User:
                    with self.session() as session:
                        return session.query(User).filter(User.name == name).first()

                def get_user_by_id(self, id: int) -> User:
                    with self.session() as session:
                        return User.get(id)

                def create_user(self, name: str) -> User:
                    return User.create(**{"name": name})
    Args:
        db_manager (DatabaseManager, optional): 数据库管理器。默认为 None。
            如果为 None，则使用默认的数据库管理器(db)。
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        """创建 BaseDao 实例。"""
        self._db_manager = db_manager or db

    def get_raw_session(self) -> Session:
        """获取一个原始的 session 对象。

        您应该手动提交或回滚 session。
        我们建议您使用 :meth:`session` 方法。

        Example:
            .. code-block:: python

                user = User(name="Edward Snowden")
                session = self.get_raw_session()
                session.add(user)
                session.commit()
                session.close()
        """
        return self._db_manager._session()  # type: ignore

    @contextmanager
    def session(self, commit: Optional[bool] = True) -> Iterator[Session]:
        """提供一个围绕一系列操作的事务范围。

        如果引发异常，则会自动回滚会话，否则将提交会话。

        Example:
            .. code-block:: python

                with self.session() as session:
                    session.query(User).filter(User.name == "Edward Snowden").first()

        Args:
            commit (Optional[bool], optional): 是否提交会话。默认为 True。

        Returns:
            Session: 一个 session 对象。

        Raises:
            Exception: 任何异常都会被抛出。
        """
        with self._db_manager.session(commit=commit) as session:
            yield session
    def from_request(self, request: QUERY_SPEC) -> T:
        """Convert a request schema object to an entity object.

        Args:
            request (REQ): The request schema object or dict for query.

        Returns:
            T: The entity object.
        """
        raise NotImplementedError

    def to_request(self, entity: T) -> REQ:
        """Convert an entity object to a request schema object.

        Args:
            entity (T): The entity object.

        Returns:
            REQ: The request schema object.
        """
        raise NotImplementedError

    def from_response(self, response: RES) -> T:
        """Convert a response schema object to an entity object.

        Args:
            response (RES): The response schema object.

        Returns:
            T: The entity object.
        """
        raise NotImplementedError

    def to_response(self, entity: T) -> RES:
        """Convert an entity object to a response schema object.

        Args:
            entity (T): The entity object.

        Returns:
            RES: The response schema object.
        """
        raise NotImplementedError

    def create(self, request: REQ) -> RES:
        """Create an entity object.

        Args:
            request (REQ): The request schema object.

        Returns:
            RES: The response schema object.
        """
        # Convert the request object to an entity object
        entry = self.from_request(request)
        
        # Open a session with commit=False to add the entry
        with self.session(commit=False) as session:
            session.add(entry)  # Add the entry to the session
            req = self.to_request(entry)  # Convert the entry back to a request schema object
            session.commit()  # Commit the transaction
            res = self.get_one(req)  # Get the response object for the created entity
            return res  # type: ignore

    def update(self, query_request: QUERY_SPEC, update_request: REQ) -> RES:
        """Update an entity object.

        Args:
            query_request (REQ): The request schema object or dict for query.
            update_request (REQ): The request schema object for update.

        Returns:
            RES: The response schema object.
        """
        # Open a session to work with database operations
        with self.session() as session:
            query = self._create_query_object(session, query_request)  # Create a query object based on query_request
            entry = query.first()  # Retrieve the first entry matching the query
            
            if entry is None:
                raise Exception("Invalid request")  # Raise exception if no entry found
            
            # Update entry attributes from update_request
            for key, value in model_to_dict(update_request).items():  # type: ignore
                if value is not None:
                    setattr(entry, key, value)  # Set attribute 'key' of entry to 'value'
            
            session.merge(entry)  # Merge updated entry into the session
            res = self.get_one(self.to_request(entry))  # Convert entry back to request schema and retrieve response
            
            if not res:
                raise Exception("Update failed")  # Raise exception if update failed
            
            return res  # Return the response object
    def delete(self, query_request: QUERY_SPEC) -> None:
        """Delete an entity object.

        Args:
            query_request (REQ): The request schema object or dict for query.
        """
        # 使用数据库会话管理器，开始一个数据库会话
        with self.session() as session:
            # 调用 _get_entity_list 方法，获取符合查询条件的实体列表
            result_list = self._get_entity_list(session, query_request)
            # 如果返回的结果列表不恰好只有一个元素，抛出数值错误异常
            if len(result_list) != 1:
                raise ValueError(
                    f"Delete request should return one result, but got "
                    f"{len(result_list)}"
                )
            # 删除第一个符合条件的实体对象
            session.delete(result_list[0])

    def get_one(self, query_request: QUERY_SPEC) -> Optional[RES]:
        """Get an entity object.

        Args:
            query_request (REQ): The request schema object or dict for query.

        Returns:
            Optional[RES]: The response schema object.
        """
        # 使用数据库会话管理器，开始一个数据库会话
        with self.session() as session:
            # 创建查询对象
            query = self._create_query_object(session, query_request)
            # 获取查询结果的第一个对象
            result = query.first()
            # 如果结果为 None，则返回 None
            if result is None:
                return None
            # 将查询结果转换为响应对象并返回
            return self.to_response(result)

    def get_list(self, query_request: QUERY_SPEC) -> List[RES]:
        """Get a list of entity objects.

        Args:
            query_request (REQ): The request schema object or dict for query.
        Returns:
            List[RES]: The response schema object.
        """
        # 使用数据库会话管理器，开始一个数据库会话
        with self.session() as session:
            # 调用 _get_entity_list 方法，获取符合查询条件的实体列表
            result_list = self._get_entity_list(session, query_request)
            # 将每个实体对象转换为响应对象，构成响应对象列表并返回
            return [self.to_response(item) for item in result_list]

    def _get_entity_list(self, session: Session, query_request: QUERY_SPEC) -> List[T]:
        """Get a list of entity objects.

        Args:
            session (Session): The session object.
            query_request (REQ): The request schema object or dict for query.
        Returns:
            List[RES]: The response schema object.
        """
        # 创建查询对象
        query = self._create_query_object(session, query_request)
        # 获取查询结果的所有对象
        result_list = query.all()
        # 返回查询结果列表
        return result_list

    def get_list_page(
        self, query_request: QUERY_SPEC, page: int, page_size: int
    ) -> PaginationResult[RES]:
        """
        Get a page of entity objects.

        Args:
            query_request (REQ): The request schema object or dict for query.
            page (int): The page number.
            page_size (int): The page size.

        Returns:
            PaginationResult: The pagination result.
        """
        with self.session() as session:
            # 使用 session 上下文管理器创建会话
            query = self._create_query_object(session, query_request)
            # 计算查询结果总数
            total_count = query.count()
            # 根据页数和页大小偏移查询结果集
            items = query.offset((page - 1) * page_size).limit(page_size)
            # 将查询结果映射为响应对象列表
            res_items = [self.to_response(item) for item in items]
            # 计算总页数
            total_pages = (total_count + page_size - 1) // page_size

            # 返回分页结果对象
            return PaginationResult(
                items=res_items,
                total_count=total_count,
                total_pages=total_pages,
                page=page,
                page_size=page_size,
            )

    def _create_query_object(
        self, session: Session, query_request: QUERY_SPEC
    ) -> BaseQuery:
        """
        Create a query object.

        Args:
            session (Session): The session object.
            query_request (QUERY_SPEC): The request schema object or dict for query.

        Returns:
            BaseQuery: The query object.
        """
        # 获取请求对象的模型类
        model_cls = type(self.from_request(query_request))
        # 根据模型类创建查询对象
        query = session.query(model_cls)
        # 将请求对象转换为字典形式
        query_dict = (
            query_request
            if isinstance(query_request, dict)
            else model_to_dict(query_request)
        )
        # 根据请求字典动态过滤查询条件
        for key, value in query_dict.items():
            if value is not None:
                if isinstance(value, (list, tuple, dict, set)):
                    continue
                query = query.filter(getattr(model_cls, key) == value)
        return query  # 返回过滤后的查询对象
```