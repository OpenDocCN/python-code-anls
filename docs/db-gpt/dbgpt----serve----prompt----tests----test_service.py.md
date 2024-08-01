# `.\DB-GPT-src\dbgpt\serve\prompt\tests\test_service.py`

```py
# 导入所需模块和函数库
from typing import List
import pytest
from dbgpt.component import SystemApp
from dbgpt.serve.core.tests.conftest import system_app
from dbgpt.storage.metadata import db
from ..api.schemas import ServeRequest, ServerResponse
from ..models.models import ServeEntity
from ..service.service import Service

# 设置和清理测试环境的 pytest fixture
@pytest.fixture(autouse=True)
def setup_and_teardown():
    # 初始化内存中的 SQLite 数据库
    db.init_db("sqlite:///:memory:")
    # 创建数据库表结构
    db.create_all()
    # yield 之前的代码部分在测试开始前执行，之后的代码在测试结束后执行

    yield

# 定义一个 pytest fixture，返回初始化后的 Service 实例
@pytest.fixture
def service(system_app: SystemApp):
    # 创建 Service 实例
    instance = Service(system_app)
    # 初始化 Service 实例
    instance.init_app(system_app)
    return instance

# 定义一个 pytest fixture，返回默认的实体字典
@pytest.fixture
def default_entity_dict():
    return {
        "chat_scene": "chat_data",
        "sub_chat_scene": "excel",
        "prompt_type": "common",
        "prompt_name": "my_prompt_1",
        "content": "Write a qsort function in python.",
        "user_name": "zhangsan",
        "sys_code": "dbgpt",
    }

# 使用 pytest.mark.parametrize 装饰器进行参数化测试，间接使用 system_app fixture
@pytest.mark.parametrize(
    "system_app",
    [{"app_config": {"DEBUG": True, "dbgpt.serve.test_key": "hello"}}],
    indirect=True,
)
def test_config_exists(service: Service):
    # 获取 Service 实例中的 SystemApp 对象
    system_app: SystemApp = service._system_app
    # 断言 DEBUG 配置为 True
    assert system_app.config.get("DEBUG") is True
    # 断言 dbgpt.serve.test_key 配置为 "hello"
    assert system_app.config.get("dbgpt.serve.test_key") == "hello"
    # 断言 Service 实例的配置属性不为 None
    assert service.config is not None

# 使用 pytest.mark.parametrize 装饰器进行参数化测试，间接使用 system_app fixture
@pytest.mark.parametrize(
    "system_app",
    [
        {
            "app_config": {
                "DEBUG": True,
                "dbgpt.serve.prompt.default_user": "dbgpt",
                "dbgpt.serve.prompt.default_sys_code": "dbgpt",
            }
        }
    ],
    indirect=True,
)
def test_config_default_user(service: Service):
    # 获取 Service 实例中的 SystemApp 对象
    system_app: SystemApp = service._system_app
    # 断言 DEBUG 配置为 True
    assert system_app.config.get("DEBUG") is True
    # 断言 dbgpt.serve.prompt.default_user 配置为 "dbgpt"
    assert system_app.config.get("dbgpt.serve.prompt.default_user") == "dbgpt"
    # 断言 Service 实例的配置属性不为 None
    assert service.config is not None
    # 断言 Service 实例的 default_user 配置为 "dbgpt"
    assert service.config.default_user == "dbgpt"
    # 断言 Service 实例的 default_sys_code 配置为 "dbgpt"
    assert service.config.default_sys_code == "dbgpt"

# 测试 Service 类的创建功能
def test_service_create(service: Service, default_entity_dict):
    # 调用 Service 实例的 create 方法创建一个实体对象
    entity: ServerResponse = service.create(ServeRequest(**default_entity_dict))
    # 使用数据库会话验证实体是否正确保存
    with db.session() as session:
        # 从数据库中获取与 entity.id 对应的 ServeEntity 对象
        db_entity: ServeEntity = session.get(ServeEntity, entity.id)
        # 逐个断言 ServeEntity 对象的属性值是否正确
        assert db_entity.id == entity.id
        assert db_entity.chat_scene == "chat_data"
        assert db_entity.sub_chat_scene == "excel"
        assert db_entity.prompt_type == "common"
        assert db_entity.prompt_name == "my_prompt_1"
        assert db_entity.content == "Write a qsort function in python."
        assert db_entity.user_name == "zhangsan"
        assert db_entity.sys_code == "dbgpt"
        assert db_entity.gmt_created is not None
        assert db_entity.gmt_modified is not None

# 测试 Service 类的更新功能
def test_service_update(service: Service, default_entity_dict):
    # 调用 Service 实例的 create 方法创建一个实体对象
    service.create(ServeRequest(**default_entity_dict))
    # 调用 Service 实例的 update 方法更新实体对象
    entity: ServerResponse = service.update(ServeRequest(**default_entity_dict))
    # 使用数据库会话对象进行操作
    with db.session() as session:
        # 从数据库中获取指定 ServeEntity 实体的记录
        db_entity: ServeEntity = session.get(ServeEntity, entity.id)
        # 断言确保从数据库中获取的实体 ID 与传入实体的 ID 相符
        assert db_entity.id == entity.id
        # 断言确保实体的 chat_scene 字段为 "chat_data"
        assert db_entity.chat_scene == "chat_data"
        # 断言确保实体的 sub_chat_scene 字段为 "excel"
        assert db_entity.sub_chat_scene == "excel"
        # 断言确保实体的 prompt_type 字段为 "common"
        assert db_entity.prompt_type == "common"
        # 断言确保实体的 prompt_name 字段为 "my_prompt_1"
        assert db_entity.prompt_name == "my_prompt_1"
        # 断言确保实体的 content 字段为 "Write a qsort function in python."
        assert db_entity.content == "Write a qsort function in python."
        # 断言确保实体的 user_name 字段为 "zhangsan"
        assert db_entity.user_name == "zhangsan"
        # 断言确保实体的 sys_code 字段为 "dbgpt"
        assert db_entity.sys_code == "dbgpt"
        # 断言确保实体的 gmt_created 字段不为空
        assert db_entity.gmt_created is not None
        # 断言确保实体的 gmt_modified 字段不为空
        assert db_entity.gmt_modified is not None
# 测试服务的获取功能，使用默认实体字典
def test_service_get(service: Service, default_entity_dict):
    # 创建一个服务请求并调用服务的创建方法
    service.create(ServeRequest(**default_entity_dict))
    # 调用服务的获取方法并获取返回的实体对象
    entity: ServerResponse = service.get(ServeRequest(**default_entity_dict))
    # 在数据库会话中获取对应的实体对象
    with db.session() as session:
        db_entity: ServeEntity = session.get(ServeEntity, entity.id)
        # 断言数据库中的实体 ID 与返回实体对象的 ID 相等
        assert db_entity.id == entity.id
        # 断言实体对象的聊天场景为 "chat_data"
        assert db_entity.chat_scene == "chat_data"
        # 断言实体对象的子聊天场景为 "excel"
        assert db_entity.sub_chat_scene == "excel"
        # 断言实体对象的提示类型为 "common"
        assert db_entity.prompt_type == "common"
        # 断言实体对象的提示名称为 "my_prompt_1"
        assert db_entity.prompt_name == "my_prompt_1"
        # 断言实体对象的内容为 "Write a qsort function in python."
        assert db_entity.content == "Write a qsort function in python."
        # 断言实体对象的用户名为 "zhangsan"
        assert db_entity.user_name == "zhangsan"
        # 断言实体对象的系统代码为 "dbgpt"
        assert db_entity.sys_code == "dbgpt"
        # 断言实体对象的创建时间不为空
        assert db_entity.gmt_created is not None
        # 断言实体对象的修改时间不为空

def test_service_delete(service: Service, default_entity_dict):
    # 创建一个服务请求并调用服务的创建方法
    service.create(ServeRequest(**default_entity_dict))
    # 调用服务的删除方法
    service.delete(ServeRequest(**default_entity_dict))
    # 调用服务的获取方法并获取返回的实体对象
    entity: ServerResponse = service.get(ServeRequest(**default_entity_dict))
    # 断言获取的实体对象为空
    assert entity is None

def test_service_get_list(service: Service):
    # 循环创建三个服务请求，每个请求都使用不同的提示名称和系统代码
    for i in range(3):
        service.create(
            ServeRequest(**{"prompt_name": f"prompt_{i}", "sys_code": "dbgpt"})
        )
    # 调用服务的获取列表方法，获取包含特定系统代码的所有实体对象
    entities: List[ServerResponse] = service.get_list(ServeRequest(sys_code="dbgpt"))
    # 断言返回的实体对象列表长度为 3
    assert len(entities) == 3
    # 遍历返回的实体对象列表并逐个断言系统代码和提示名称符合预期
    for i, entity in enumerate(entities):
        assert entity.sys_code == "dbgpt"
        assert entity.prompt_name == f"prompt_{i}"

def test_service_get_list_by_page(service: Service):
    # 循环创建三个服务请求，每个请求都使用不同的提示名称和系统代码
    for i in range(3):
        service.create(
            ServeRequest(**{"prompt_name": f"prompt_{i}", "sys_code": "dbgpt"})
        )
    # 调用服务的分页获取列表方法，获取第一页、每页两个实体对象的结果
    res = service.get_list_by_page(ServeRequest(sys_code="dbgpt"), page=1, page_size=2)
    # 断言返回结果不为空
    assert res is not None
    # 断言返回结果的总数为 3
    assert res.total_count == 3
    # 断言返回结果的总页数为 2
    assert res.total_pages == 2
    # 断言返回结果中包含的实体对象列表长度为 2
    assert len(res.items) == 2
    # 遍历返回结果中的实体对象列表并逐个断言系统代码和提示名称符合预期
    for i, entity in enumerate(res.items):
        assert entity.sys_code == "dbgpt"
        assert entity.prompt_name == f"prompt_{i}"
```