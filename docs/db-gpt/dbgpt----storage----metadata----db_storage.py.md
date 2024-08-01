# `.\DB-GPT-src\dbgpt\storage\metadata\db_storage.py`

```py
"""Database storage implementation using SQLAlchemy."""
# 导入必要的模块和类
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional, Type, Union

# 导入 SQLAlchemy 相关模块
from sqlalchemy import URL
from sqlalchemy.orm import DeclarativeMeta, Session

# 导入自定义模块和类
from dbgpt.core import Serializer
from dbgpt.core.interface.storage import (
    QuerySpec,
    ResourceIdentifier,
    StorageInterface,
    StorageItemAdapter,
    T,
)
# 导入本地的数据库管理类
from .db_manager import BaseModel, BaseQuery, DatabaseManager


def _copy_public_properties(src: BaseModel, dest: BaseModel):
    """Copy public properties from src to dest."""
    # 遍历源对象的列，将非"id"列的属性复制到目标对象
    for column in src.__table__.columns:  # type: ignore
        if column.name != "id":
            value = getattr(src, column.name)
            if value is not None:
                setattr(dest, column.name, value)


class SQLAlchemyStorage(StorageInterface[T, BaseModel]):
    """Database storage implementation using SQLAlchemy."""

    def __init__(
        self,
        db_url_or_db: Union[str, URL, DatabaseManager],
        model_class: Type[BaseModel],
        adapter: StorageItemAdapter[T, BaseModel],
        serializer: Optional[Serializer] = None,
        engine_args: Optional[Dict] = None,
        base: Optional[DeclarativeMeta] = None,
        query_class=BaseQuery,
    ):
        """Create a SQLAlchemyStorage instance."""
        # 调用父类的初始化方法，设置序列化器和适配器
        super().__init__(serializer=serializer, adapter=adapter)
        # 创建数据库管理器对象
        self.db_manager = DatabaseManager.build_from(
            db_url_or_db, engine_args, base, query_class
        )
        # 设置模型类
        self._model_class = model_class

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Return a session."""
        # 使用数据库管理器创建会话上下文
        with self.db_manager.session() as session:
            yield session

    def save(self, data: T) -> None:
        """Save data to the storage."""
        # 保存数据到数据库
        with self.session() as session:
            model_instance = self.adapter.to_storage_format(data)
            session.add(model_instance)

    def update(self, data: T) -> None:
        """Update data in the storage."""
        # 更新数据库中的数据
        with self.session() as session:
            # 获取用于查询的查询对象
            query = self.adapter.get_query_for_identifier(
                self._model_class, data.identifier, session=session
            )
            # 查询数据库中是否存在指定的模型实例
            exist_model_instance = query.with_session(session).first()
            if exist_model_instance:
                # 如果存在，将新数据的公共属性复制到已存在的实例中
                _copy_public_properties(
                    self.adapter.to_storage_format(data), exist_model_instance
                )
                # 合并更新到数据库中
                session.merge(exist_model_instance)
                return
    def save_or_update(self, data: T) -> None:
        """Save or update data in the storage."""
        # 开启一个数据库会话
        with self.session() as session:
            # 获取用于查询指定标识符的查询对象
            query = self.adapter.get_query_for_identifier(
                self._model_class, data.identifier, session=session
            )
            # 在当前会话中执行查询，获取模型实例
            model_instance = query.with_session(session).first()
            # 如果找到了模型实例
            if model_instance:
                # 将传入的数据转换为存储格式，并复制公共属性到模型实例中
                new_instance = self.adapter.to_storage_format(data)
                _copy_public_properties(new_instance, model_instance)
                # 将更新后的模型实例合并回数据库会话中
                session.merge(model_instance)
                return
        # 如果没有找到对应的模型实例，则保存新数据
        self.save(data)

    def load(self, resource_id: ResourceIdentifier, cls: Type[T]) -> Optional[T]:
        """Load data by identifier from the storage."""
        # 开启一个数据库会话
        with self.session() as session:
            # 获取用于查询指定标识符的查询对象
            query = self.adapter.get_query_for_identifier(
                self._model_class, resource_id, session=session
            )
            # 在当前会话中执行查询，获取模型实例
            model_instance = query.with_session(session).first()
            # 如果找到了模型实例
            if model_instance:
                # 将数据库中的模型实例转换为业务逻辑对象并返回
                return self.adapter.from_storage_format(model_instance)
            # 如果未找到对应的模型实例，则返回 None
            return None

    def delete(self, resource_id: ResourceIdentifier) -> None:
        """Delete data by identifier from the storage."""
        # 开启一个数据库会话
        with self.session() as session:
            # 获取用于查询指定标识符的查询对象
            query = self.adapter.get_query_for_identifier(
                self._model_class, resource_id, session=session
            )
            # 在当前会话中执行查询，获取模型实例
            model_instance = query.with_session(session).first()
            # 如果找到了模型实例
            if model_instance:
                # 删除数据库中的模型实例
                session.delete(model_instance)

    def query(self, spec: QuerySpec, cls: Type[T]) -> List[T]:
        """Query data from the storage.

        Args:
            spec (QuerySpec): The query specification
            cls (Type[T]): The type of the data
        """
        # 开启一个数据库会话
        with self.session() as session:
            # 创建查询对象，查询指定类型的所有模型实例
            query = session.query(self._model_class)
            # 根据查询规范中的条件筛选模型实例
            for key, value in spec.conditions.items():
                if value is not None:
                    query = query.filter(getattr(self._model_class, key) == value)
            # 如果查询规范中有限制条件，则设置查询结果的最大返回数
            if spec.limit is not None:
                query = query.limit(spec.limit)
            # 如果查询规范中有偏移条件，则设置查询结果的偏移量
            if spec.offset is not None:
                query = query.offset(spec.offset)
            # 执行查询，获取所有符合条件的模型实例列表，并将其转换为业务逻辑对象列表后返回
            model_instances = query.all()
            return [
                self.adapter.from_storage_format(instance)
                for instance in model_instances
            ]

    def count(self, spec: QuerySpec, cls: Type[T]) -> int:
        """Count the number of data in the storage.

        Args:
            spec (QuerySpec): The query specification
            cls (Type[T]): The type of the data
        """
        # 开启一个数据库会话
        with self.session() as session:
            # 创建查询对象，查询指定类型的所有模型实例
            query = session.query(self._model_class)
            # 根据查询规范中的条件筛选模型实例
            for key, value in spec.conditions.items():
                if value is not None:
                    query = query.filter(getattr(self._model_class, key) == value)
            # 返回符合条件的模型实例数量
            return query.count()
```