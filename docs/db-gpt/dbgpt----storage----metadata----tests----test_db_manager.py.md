# `.\DB-GPT-src\dbgpt\storage\metadata\tests\test_db_manager.py`

```py
from __future__ import annotations

import tempfile
from typing import Type

import pytest
from sqlalchemy import Column, Integer, String

from dbgpt.storage.metadata.db_manager import (
    BaseModel,
    DatabaseManager,
    PaginationResult,
    create_model,
)


@pytest.fixture
def db():
    # 创建一个内存中的 SQLite 数据库管理器实例
    db = DatabaseManager()
    db.init_db("sqlite:///:memory:")
    return db


@pytest.fixture
def Model(db):
    # 使用数据库管理器创建一个模型，并返回该模型类
    return create_model(db)


def test_database_initialization(db: DatabaseManager, Model: Type[BaseModel]):
    # 断言数据库引擎和会话对象不为空
    assert db.engine is not None
    assert db.session is not None

    with db.session() as session:
        # 断言数据库会话对象不为空
        assert session is not None


def test_model_creation(db: DatabaseManager, Model: Type[BaseModel]):
    # 断言数据库的表格字典为空
    assert db.metadata.tables == {}

    # 定义一个名为 User 的模型类，包含 id 和 name 两个列
    class User(Model):
        __tablename__ = "user"
        id = Column(Integer, primary_key=True)
        name = Column(String(50))

    db.create_all()
    # 断言数据库的表格字典的第一个键名为 "user"
    assert list(db.metadata.tables.keys())[0] == "user"


def test_crud_operations(db: DatabaseManager, Model: Type[BaseModel]):
    # 定义一个名为 User 的模型类，包含 id 和 name 两个列
    class User(Model):
        __tablename__ = "user"
        id = Column(Integer, primary_key=True)
        name = Column(String(50))

    db.create_all()

    # Create
    with db.session() as session:
        # 在数据库会话中创建一个名为 "John Doe" 的 User 对象
        user = User(name="John Doe")
        session.add(user)

    # Read
    with db.session() as session:
        # 查询名为 "John Doe" 的 User 对象
        user = session.query(User).filter_by(name="John Doe").first()
        assert user is not None

    # Update
    with db.session() as session:
        # 查询名为 "John Doe" 的 User 对象并修改其名称为 "Mike Doe"
        user = session.query(User).filter_by(name="John Doe").first()
        user.name = "Mike Doe"
        session.merge(user)
    with db.session() as session:
        # 断言名为 "Mike Doe" 的 User 对象存在，而名为 "John Doe" 的对象不存在
        user = session.query(User).filter_by(name="Mike Doe").first()
        assert user is not None
        assert session.query(User).filter(User.name == "John Doe").first() is None

    # Delete
    with db.session() as session:
        # 查询名为 "Mike Doe" 的 User 对象并删除
        user = session.query(User).filter_by(name="Mike Doe").first()
        session.delete(user)

    with db.session() as session:
        # 断言数据库中的 User 表格为空
        assert len(session.query(User).all()) == 0


def test_crud_mixins(db: DatabaseManager, Model: Type[BaseModel]):
    # 定义一个名为 User 的模型类，包含 id 和 name 两个列
    class User(Model):
        __tablename__ = "user"
        id = Column(Integer, primary_key=True)
        name = Column(String(50))

    db.create_all()
    # 断言 User 模型类的 db() 方法返回数据库管理器对象
    User.db() == db


def test_pagination_query(db: DatabaseManager, Model: Type[BaseModel]):
    # 定义一个名为 User 的模型类，包含 id 和 name 两个列
    class User(Model):
        __tablename__ = "user"
        id = Column(Integer, primary_key=True)
        name = Column(String(50))

    db.create_all()

    with db.session() as session:
        # 在数据库会话中添加 30 个 User 对象
        for i in range(30):
            user = User(name=f"User {i}")
            session.add(user)
    with db.session() as session:
        # 使用分页查询方法查询第一页，每页10个对象
        users_page_1 = session.query(User).paginate_query(page=1, per_page=10)
        assert len(users_page_1.items) == 10  # 断言返回的对象数量为10
        assert users_page_1.total_pages == 3  # 断言总页数为3


def test_invalid_pagination(db: DatabaseManager, Model: Type[BaseModel]):
    # 此测试函数尚未完整实现，略过注释部分
    pass
    # 定义名为 User 的数据库模型，映射到 "user" 表格
    class User(Model):
        # 表格名为 "user"
        __tablename__ = "user"
        # 主键列 id，整数类型
        id = Column(Integer, primary_key=True)
        # 名称列 name，字符串类型，最大长度为 50
        name = Column(String(50))
    
    # 创建所有数据库表格（如果不存在）
    db.create_all()
    
    # 使用 pytest 框架的 assertRaises 方法检查是否抛出 ValueError 异常
    # 第一个测试用例：检查在页码为 0 时调用 paginate_query 是否会引发异常
    with pytest.raises(ValueError):
        # 使用 db.session() 方法创建数据库会话
        with db.session() as session:
            # 查询 User 模型，并尝试分页查询，页码为 0，每页 10 条记录
            session.query(User).paginate_query(page=0, per_page=10)
    
    # 第二个测试用例：检查在每页显示数为负数时调用 paginate_query 是否会引发异常
    with pytest.raises(ValueError):
        # 使用 db.session() 方法创建数据库会话
        with db.session() as session:
            # 查询 User 模型，并尝试分页查询，页码为 1，每页显示数为 -1
            session.query(User).paginate_query(page=1, per_page=-1)
# 确保传入的数据库管理器中没有已存在的表格
def test_set_model_db_manager(db: DatabaseManager, Model: Type[BaseModel]):
    assert db.metadata.tables == {}

    # 定义一个新的模型类，继承自传入的 Model 类，并指定表名和字段
    class User(Model):
        __tablename__ = "user"
        id = Column(Integer, primary_key=True)
        name = Column(String(50))

    # 使用临时文件创建一个 SQLite 数据库作为测试数据库
    with tempfile.NamedTemporaryFile(delete=True) as db_file:
        filename = db_file.name
        # 使用 DatabaseManager.build_from 方法创建一个新的数据库管理器
        new_db = DatabaseManager.build_from(
            f"sqlite:///{filename}", base=Model, override_query_class=True
        )
        # 将模型类关联到新的数据库管理器
        Model.set_db(new_db)
        # 创建模型对应的表格
        new_db.create_all()
        # 创建传入的数据库管理器中的表格
        db.create_all()
        
        # 确保新数据库管理器中有且仅有一个表格，即'user'
        assert list(new_db.metadata.tables.keys())[0] == "user"
        
        # 使用新数据库管理器创建一个会话，添加一个用户到'user'表中
        with new_db.session() as session:
            user = User(name="John Doe")
            session.add(user)
        
        # 使用新数据库管理器创建一个会话，确保能够查询到刚才添加的用户
        with new_db.session() as session:
            assert session.query(User).filter_by(name="John Doe").first() is not None
        
        # 使用传入的数据库管理器创建一个会话，确保在原数据库管理器中查询不到添加的用户
        with db.session() as session:
            assert session.query(User).filter_by(name="John Doe").first() is None
        
        # 使用新数据库管理器创建一个会话，查询'user'表中的所有记录并检查数量为1
        with new_db.session() as session:
            assert session.query(User).count() == 1
            # 查询'user'表中第一条记录的姓名，确保为'John Doe'
            assert session.query(User).filter(
                User.name == "John Doe"
            ).first().name == "John Doe"
```