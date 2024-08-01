# `.\DB-GPT-src\dbgpt\storage\metadata\tests\test_base_dao.py`

```py
from typing import Any, Dict, Optional, Type, Union  # 引入必要的类型提示

import pytest  # 引入pytest测试框架
from sqlalchemy import Column, Integer, String  # 从sqlalchemy库引入Column、Integer和String

from dbgpt._private.pydantic import BaseModel as PydanticBaseModel  # 从dbgpt._private.pydantic中引入BaseModel和Field
from dbgpt._private.pydantic import Field, model_to_dict  # 从dbgpt._private.pydantic中引入Field和model_to_dict
from dbgpt.storage.metadata.db_manager import (  # 从dbgpt.storage.metadata.db_manager中引入BaseModel、DatabaseManager、PaginationResult和create_model
    BaseModel,
    DatabaseManager,
    PaginationResult,
    create_model,
)

from .._base_dao import BaseDao  # 从相对路径中引入BaseDao

# 定义用户请求的Pydantic模型
class UserRequest(PydanticBaseModel):
    name: str = Field(..., description="User name")  # 用户名，字符串类型，必填
    age: Optional[int] = Field(default=-1, description="User age")  # 年龄，可选整数，默认为-1
    password: Optional[str] = Field(default="", description="User password")  # 密码，可选字符串，默认为空字符串

# 定义用户响应的Pydantic模型
class UserResponse(PydanticBaseModel):
    id: int = Field(..., description="User id")  # 用户ID，整数类型，必填
    name: str = Field(..., description="User name")  # 用户名，字符串类型，必填
    age: Optional[int] = Field(default=-1, description="User age")  # 年龄，可选整数，默认为-1

# 定义数据库的fixture
@pytest.fixture
def db():
    db = DatabaseManager()  # 创建DatabaseManager对象
    db.init_db("sqlite:///:memory:")  # 在内存中初始化数据库
    return db

# 定义Model的fixture，创建数据库模型
@pytest.fixture
def Model(db):
    return create_model(db)  # 使用db创建数据库模型

# 定义User的fixture，创建用户模型
@pytest.fixture
def User(Model):
    class User(Model):
        __tablename__ = "user"  # 数据库表名为'user'
        id = Column(Integer, primary_key=True)  # 主键，整数类型
        name = Column(String(50), unique=True)  # 唯一用户名，字符串类型，长度50
        age = Column(Integer)  # 年龄，整数类型
        password = Column(String(50))  # 密码，字符串类型，长度50

    return User

# 定义user_req的fixture，创建用户请求对象
@pytest.fixture
def user_req():
    return UserRequest(name="Edward Snowden", age=30, password="123456")  # 创建UserRequest对象

# 定义user_dao的fixture，创建用户数据访问对象
@pytest.fixture
def user_dao(db, User):
    class UserDao(BaseDao[User, UserRequest, UserResponse]):
        # 从请求中创建用户对象
        def from_request(self, request: Union[UserRequest, Dict[str, Any]]) -> User:
            if isinstance(request, UserRequest):
                return User(**model_to_dict(request))
            else:
                return User(**request)

        # 转换用户对象为请求对象
        def to_request(self, entity: User) -> UserRequest:
            return UserRequest(
                name=entity.name, age=entity.age, password=entity.password
            )

        # 从响应中创建用户对象
        def from_response(self, response: UserResponse) -> User:
            return User(**model_to_dict(response))

        # 转换用户对象为响应对象
        def to_response(self, entity: User):
            return UserResponse(id=entity.id, name=entity.name, age=entity.age)

    db.create_all()  # 创建所有表
    return UserDao(db)

# 测试创建用户的函数
def test_create_user(db: DatabaseManager, User: Type[BaseModel], user_dao, user_req):
    user_dao.create(user_req)  # 创建用户
    with db.session() as session:
        user = session.query(User).first()  # 查询第一个用户
        assert user.name == user_req.name  # 断言用户的姓名与请求中的姓名一致
        assert user.age == user_req.age  # 断言用户的年龄与请求中的年龄一致
        assert user.password == user_req.password  # 断言用户的密码与请求中的密码一致

# 测试更新用户的函数
def test_update_user(db: DatabaseManager, User: Type[BaseModel], user_dao, user_req):
    # 创建用户
    created_user_response = user_dao.create(user_req)

    # 更新用户信息
    updated_req = UserRequest(name=user_req.name, age=35, password="newpassword")
    updated_user = user_dao.update(
        query_request={"name": user_req.name}, update_request=updated_req
    )
    assert updated_user.id == created_user_response.id  # 断言更新后的用户ID与创建时的用户ID一致
    assert updated_user.age == 35  # 断言更新后的用户年龄为35
    # 使用数据库会话对象打开一个会话上下文
    with db.session() as session:
        # 从数据库中获取指定 ID 的用户对象
        user = session.get(User, created_user_response.id)
        # 断言获取的用户对象的年龄为 35，用于验证用户信息是否正确更新
        assert user.age == 35
# 定义一个测试函数，用于部分更新用户信息
def test_update_user_partial(
    db: DatabaseManager, User: Type[BaseModel], user_dao, user_req
):
    # 创建一个用户并获取创建响应
    created_user_response = user_dao.create(user_req)

    # 更新用户信息，设置新密码并清除年龄
    updated_req = UserRequest(name=user_req.name, password="newpassword")
    updated_req.age = None
    updated_user = user_dao.update(
        query_request={"name": user_req.name}, update_request=updated_req
    )
    # 断言更新后用户的ID与创建时相同，年龄与原请求中的年龄相同
    assert updated_user.id == created_user_response.id
    assert updated_user.age == user_req.age

    # 在数据库中验证用户是否更新
    with db.session() as session:
        user = session.get(User, created_user_response.id)
        assert user.age == user_req.age
        assert user.password == "newpassword"


# 定义一个测试函数，用于获取用户信息
def test_get_user(db: DatabaseManager, User: Type[BaseModel], user_dao, user_req):
    # 创建一个用户并获取创建响应
    created_user_response = user_dao.create(user_req)

    # 查询用户信息
    fetched_user = user_dao.get_one({"name": user_req.name})
    # 断言查询到的用户ID与创建时相同，姓名与原请求中的姓名相同，年龄与原请求中的年龄相同
    assert fetched_user.id == created_user_response.id
    assert fetched_user.name == user_req.name
    assert fetched_user.age == user_req.age


# 定义一个测试函数，用于获取符合条件的用户列表
def test_get_list_user(db: DatabaseManager, User: Type[BaseModel], user_dao):
    # 循环创建20个用户
    for i in range(20):
        user_dao.create(
            UserRequest(
                name=f"User {i}", age=i, password="123456" if i % 2 == 0 else "abcdefg"
            )
        )
    # 查询符合条件的用户列表
    fetched_user = user_dao.get_list({"password": "123456"})
    # 断言查询到的用户数量为10
    assert len(fetched_user) == 10


# 定义一个测试函数，用于分页获取符合条件的用户列表
def test_get_list_page_user(db: DatabaseManager, User: Type[BaseModel], user_dao):
    # 循环创建20个用户
    for i in range(20):
        user_dao.create(
            UserRequest(
                name=f"User {i}", age=i, password="123456" if i % 2 == 0 else "abcdefg"
            )
        )
    # 分页查询符合条件的用户列表，获取第一页
    page_result: PaginationResult = user_dao.get_list_page(
        {"password": "123456"}, page=1, page_size=3
    )
    # 断言总用户数为10，总页数为4，当前页面用户数为3，第一个用户姓名为"User 0"
    assert page_result.total_count == 10
    assert page_result.total_pages == 4
    assert len(page_result.items) == 3
    assert page_result.items[0].name == "User 0"

    # 测试查询下一页
    # 分页查询符合条件的用户列表，获取第二页
    page_result: PaginationResult = user_dao.get_list_page(
        {"password": "123456"}, page=2, page_size=3
    )
    # 断言总用户数为10，总页数为4，当前页面用户数为3，第一个用户姓名为"User 6"
    assert page_result.total_count == 10
    assert page_result.total_pages == 4
    assert len(page_result.items) == 3
    assert page_result.items[0].name == "User 6"
```