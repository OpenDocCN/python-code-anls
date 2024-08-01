# `.\DB-GPT-src\dbgpt\serve\prompt\tests\test_models.py`

```py
# 导入所需模块和库
from typing import List
import pytest
from dbgpt.storage.metadata import db
from ..api.schemas import ServeRequest, ServerResponse
from ..config import ServeConfig
from ..models.models import ServeDao, ServeEntity

# 定义一个自动使用的测试夹具，用于设置和清理测试环境
@pytest.fixture(autouse=True)
def setup_and_teardown():
    # 初始化内存中的 SQLite 数据库
    db.init_db("sqlite:///:memory:")
    # 创建所有数据库表结构
    db.create_all()
    # 使用 yield 表示在测试运行前执行和测试结束后执行的代码段
    yield

# 定义一个测试夹具，返回 ServeConfig 的实例
@pytest.fixture
def server_config():
    return ServeConfig()

# 定义一个测试夹具，返回 ServeDao 对象的实例
@pytest.fixture
def dao(server_config):
    return ServeDao(server_config)

# 定义一个测试夹具，返回默认的实体字典
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
        "prompt_language": "zh",
        "model": "vicuna-13b-v1.5",
    }

# 测试函数：验证 ServeEntity 表在数据库元数据中存在
def test_table_exist():
    assert ServeEntity.__tablename__ in db.metadata.tables

# 测试函数：创建实体并验证其属性值
def test_entity_create(default_entity_dict):
    with db.session() as session:
        entity: ServeEntity = ServeEntity(**default_entity_dict)
        session.add(entity)
        session.commit()
        db_entity: ServeEntity = session.get(ServeEntity, entity.id)
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

# 测试函数：验证实体的唯一键约束
def test_entity_unique_key(default_entity_dict):
    with db.session() as session:
        entity = ServeEntity(**default_entity_dict)
        session.add(entity)
    # 使用 pytest.raises 检查在插入重复唯一键时是否抛出异常
    with pytest.raises(Exception):
        with db.session() as session:
            entity = ServeEntity(
                **{
                    "prompt_name": "my_prompt_1",
                    "sys_code": "dbgpt",
                    "prompt_language": "zh",
                    "model": "vicuna-13b-v1.5",
                }
            )
            session.add(entity)

# 测试函数：测试获取实体
def test_entity_get(default_entity_dict):
    # 使用数据库会话对象开始一个事务
    with db.session() as session:
        # 创建一个新的 ServeEntity 实例，并使用默认实体字典初始化
        entity = ServeEntity(**default_entity_dict)
        # 将新实体添加到当前会话中
        session.add(entity)
        # 提交事务，将新实体持久化到数据库
        session.commit()
        # 从数据库中获取刚刚插入的 ServeEntity 实例
        db_entity: ServeEntity = session.get(ServeEntity, entity.id)
        # 确保数据库返回的实体 ID 与插入时的 ID 相同
        assert db_entity.id == entity.id
        # 确保数据库返回的实体的 chat_scene 属性值为 "chat_data"
        assert db_entity.chat_scene == "chat_data"
        # 确保数据库返回的实体的 sub_chat_scene 属性值为 "excel"
        assert db_entity.sub_chat_scene == "excel"
        # 确保数据库返回的实体的 prompt_type 属性值为 "common"
        assert db_entity.prompt_type == "common"
        # 确保数据库返回的实体的 prompt_name 属性值为 "my_prompt_1"
        assert db_entity.prompt_name == "my_prompt_1"
        # 确保数据库返回的实体的 content 属性值为 "Write a qsort function in python."
        assert db_entity.content == "Write a qsort function in python."
        # 确保数据库返回的实体的 user_name 属性值为 "zhangsan"
        assert db_entity.user_name == "zhangsan"
        # 确保数据库返回的实体的 sys_code 属性值为 "dbgpt"
        assert db_entity.sys_code == "dbgpt"
        # 确保数据库返回的实体的 gmt_created 属性值不为空
        assert db_entity.gmt_created is not None
        # 确保数据库返回的实体的 gmt_modified 属性值不为空
        assert db_entity.gmt_modified is not None
# 测试函数：更新实体信息至数据库中
def test_entity_update(default_entity_dict):
    # 使用数据库会话，添加默认实体信息
    with db.session() as session:
        entity = ServeEntity(**default_entity_dict)
        session.add(entity)
        session.commit()
        # 更新实体的提示名称为"my_prompt_2"
        entity.prompt_name = "my_prompt_2"
        # 合并更新后的实体信息
        session.merge(entity)
        # 从数据库中获取更新后的实体
        db_entity: ServeEntity = session.get(ServeEntity, entity.id)
        # 断言检查数据库中实体的各个属性
        assert db_entity.id == entity.id
        assert db_entity.chat_scene == "chat_data"
        assert db_entity.sub_chat_scene == "excel"
        assert db_entity.prompt_type == "common"
        assert db_entity.prompt_name == "my_prompt_2"
        assert db_entity.content == "Write a qsort function in python."
        assert db_entity.user_name == "zhangsan"
        assert db_entity.sys_code == "dbgpt"
        assert db_entity.gmt_created is not None
        assert db_entity.gmt_modified is not None


# 测试函数：从数据库中删除实体
def test_entity_delete(default_entity_dict):
    # 使用数据库会话，添加默认实体信息
    with db.session() as session:
        entity = ServeEntity(**default_entity_dict)
        session.add(entity)
        session.commit()
        # 删除数据库中的实体
        session.delete(entity)
        session.commit()
        # 确认实体已从数据库中删除
        db_entity: ServeEntity = session.get(ServeEntity, entity.id)
        assert db_entity is None


# 测试函数：添加多个实体并查询全部实体
def test_entity_all():
    # 使用数据库会话，循环添加多个实体
    with db.session() as session:
        for i in range(10):
            entity = ServeEntity(
                chat_scene="chat_data",
                sub_chat_scene="excel",
                prompt_type="common",
                prompt_name=f"my_prompt_{i}",
                content="Write a qsort function in python.",
                user_name="zhangsan",
                sys_code="dbgpt",
            )
            session.add(entity)
    # 使用数据库会话，查询所有实体并进行断言检查
    with db.session() as session:
        entities = session.query(ServeEntity).all()
        assert len(entities) == 10
        for entity in entities:
            assert entity.chat_scene == "chat_data"
            assert entity.sub_chat_scene == "excel"
            assert entity.prompt_type == "common"
            assert entity.content == "Write a qsort function in python."
            assert entity.user_name == "zhangsan"
            assert entity.sys_code == "dbgpt"
            assert entity.gmt_created is not None
            assert entity.gmt_modified is not None


# 测试函数：使用DAO创建实体
def test_dao_create(dao, default_entity_dict):
    # 创建服务请求对象
    req = ServeRequest(**default_entity_dict)
    # 调用DAO创建实体并获取响应
    res: ServerResponse = dao.create(req)
    # 断言检查创建实体的各个属性
    assert res is not None
    assert res.id == 1
    assert res.chat_scene == "chat_data"
    assert res.sub_chat_scene == "excel"
    assert res.prompt_type == "common"
    assert res.prompt_name == "my_prompt_1"
    assert res.content == "Write a qsort function in python."
    assert res.user_name == "zhangsan"
    assert res.sys_code == "dbgpt"


# 测试函数：使用DAO获取单个实体
def test_dao_get_one(dao, default_entity_dict):
    # 创建服务请求对象
    req = ServeRequest(**default_entity_dict)
    # 调用DAO创建实体并获取响应
    res: ServerResponse = dao.create(req)
    # 调用DAO根据条件获取单个实体
    res: ServerResponse = dao.get_one(
        {"prompt_name": "my_prompt_1", "sys_code": "dbgpt"}
    )
    # 断言检查获取的实体对象不为空
    assert res is not None
    # 断言检查结果对象的 id 属性是否为 1
    assert res.id == 1
    # 断言检查结果对象的 chat_scene 属性是否为 "chat_data"
    assert res.chat_scene == "chat_data"
    # 断言检查结果对象的 sub_chat_scene 属性是否为 "excel"
    assert res.sub_chat_scene == "excel"
    # 断言检查结果对象的 prompt_type 属性是否为 "common"
    assert res.prompt_type == "common"
    # 断言检查结果对象的 prompt_name 属性是否为 "my_prompt_1"
    assert res.prompt_name == "my_prompt_1"
    # 断言检查结果对象的 content 属性是否为 "Write a qsort function in python."
    assert res.content == "Write a qsort function in python."
    # 断言检查结果对象的 user_name 属性是否为 "zhangsan"
    assert res.user_name == "zhangsan"
    # 断言检查结果对象的 sys_code 属性是否为 "dbgpt"
    assert res.sys_code == "dbgpt"
def test_get_dao_get_list(dao):
    # 循环创建10条服务请求记录
    for i in range(10):
        # 根据不同的循环索引i，创建ServeRequest对象并存入数据库
        dao.create(
            ServeRequest(
                chat_scene="chat_data",  # 对话场景为"chat_data"
                sub_chat_scene="excel",   # 子场景为"excel"
                prompt_type="common",     # 提示类型为"common"
                prompt_name=f"my_prompt_{i}",  # 提示名称为"my_prompt_i"
                content="Write a qsort function in python.",  # 内容固定为"Write a qsort function in python."
                user_name="zhangsan" if i % 2 == 0 else "lisi",  # 用户名根据i的奇偶性确定
                sys_code="dbgpt",         # 系统代码为"dbgpt"
            )
        )
    
    # 查询数据库中sys_code为"dbgpt"的所有记录
    res: List[ServerResponse] = dao.get_list({"sys_code": "dbgpt"})
    # 断言返回的记录数为10
    assert len(res) == 10
    
    # 遍历返回的记录列表res，进行详细断言
    for i, r in enumerate(res):
        assert r.id == i + 1  # 断言id与索引i+1相等（索引从0开始，id从1开始）
        assert r.chat_scene == "chat_data"  # 断言对话场景为"chat_data"
        assert r.sub_chat_scene == "excel"  # 断言子场景为"excel"
        assert r.prompt_type == "common"    # 断言提示类型为"common"
        assert r.prompt_name == f"my_prompt_{i}"  # 断言提示名称与"my_prompt_i"相等
        assert r.content == "Write a qsort function in python."  # 断言内容固定为"Write a qsort function in python."
        assert r.user_name == "zhangsan" if i % 2 == 0 else "lisi"  # 断言用户名根据i的奇偶性确定
        assert r.sys_code == "dbgpt"   # 断言系统代码为"dbgpt"
    
    # 查询数据库中user_name为"zhangsan"的所有记录
    half_res: List[ServerResponse] = dao.get_list({"user_name": "zhangsan"})
    # 断言返回的记录数为5
    assert len(half_res) == 5


def test_dao_update(dao, default_entity_dict):
    # 创建一个服务请求对象req，并存入数据库
    req = ServeRequest(**default_entity_dict)
    res: ServerResponse = dao.create(req)
    
    # 更新数据库中prompt_name为"my_prompt_1"且sys_code为"dbgpt"的记录，将其prompt_name更新为"my_prompt_2"
    res: ServerResponse = dao.update(
        {"prompt_name": "my_prompt_1", "sys_code": "dbgpt"},
        ServeRequest(prompt_name="my_prompt_2"),
    )
    
    # 断言更新操作成功
    assert res is not None
    assert res.id == 1  # 断言id为1
    assert res.chat_scene == "chat_data"  # 断言对话场景为"chat_data"
    assert res.sub_chat_scene == "excel"  # 断言子场景为"excel"
    assert res.prompt_type == "common"    # 断言提示类型为"common"
    assert res.prompt_name == "my_prompt_2"  # 断言提示名称为"my_prompt_2"
    assert res.content == "Write a qsort function in python."  # 断言内容固定为"Write a qsort function in python."
    assert res.user_name == "zhangsan"    # 断言用户名为"zhangsan"
    assert res.sys_code == "dbgpt"        # 断言系统代码为"dbgpt"


def test_dao_delete(dao, default_entity_dict):
    # 创建一个服务请求对象req，并存入数据库
    req = ServeRequest(**default_entity_dict)
    res: ServerResponse = dao.create(req)
    
    # 删除数据库中prompt_name为"my_prompt_1"且sys_code为"dbgpt"的记录
    dao.delete({"prompt_name": "my_prompt_1", "sys_code": "dbgpt"})
    
    # 查询数据库中prompt_name为"my_prompt_1"且sys_code为"dbgpt"的记录
    res: ServerResponse = dao.get_one(
        {"prompt_name": "my_prompt_1", "sys_code": "dbgpt"}
    )
    
    # 断言查询结果为空
    assert res is None


def test_dao_get_list_page(dao):
    # 循环创建20条服务请求记录
    for i in range(20):
        # 根据不同的循环索引i，创建ServeRequest对象并存入数据库
        dao.create(
            ServeRequest(
                chat_scene="chat_data",  # 对话场景为"chat_data"
                sub_chat_scene="excel",   # 子场景为"excel"
                prompt_type="common",     # 提示类型为"common"
                prompt_name=f"my_prompt_{i}",  # 提示名称为"my_prompt_i"
                content="Write a qsort function in python.",  # 内容固定为"Write a qsort function in python."
                user_name="zhangsan" if i % 2 == 0 else "lisi",  # 用户名根据i的奇偶性确定
                sys_code="dbgpt",         # 系统代码为"dbgpt"
            )
        )
    
    # 查询数据库中sys_code为"dbgpt"的记录，分页查询第1页，每页大小为8
    res = dao.get_list_page({"sys_code": "dbgpt"}, page=1, page_size=8)
    
    # 断言总记录数为20
    assert res.total_count == 20
    # 断言总页数为3
    assert res.total_pages == 3
    # 断言当前页为第1页
    assert res.page == 1
    # 断言每页大小为8
    assert res.page_size == 8
    # 断言返回的记录条数为8
    assert len(res.items) == 8
    # 遍历 res 对象的每个元素及其索引
    for i, r in enumerate(res.items):
        # 断言每个元素的 id 应与索引加一相等，确保从 1 开始递增
        assert r.id == i + 1
        # 断言 chat_scene 应为 "chat_data"，表示聊天场景为数据聊天
        assert r.chat_scene == "chat_data"
        # 断言 sub_chat_scene 应为 "excel"，表示详细的聊天子场景为 Excel
        assert r.sub_chat_scene == "excel"
        # 断言 prompt_type 应为 "common"，表示提示类型为普通
        assert r.prompt_type == "common"
        # 断言 prompt_name 应为特定格式的字符串，包含索引值
        assert r.prompt_name == f"my_prompt_{i}"
        # 断言 content 应为固定内容的字符串，提示编写 Python 中的快速排序函数
        assert r.content == "Write a qsort function in python."
        # 断言 user_name 应为 "zhangsan" 或 "lisi"，根据索引奇偶性确定
        assert r.user_name == "zhangsan" if i % 2 == 0 else "lisi"
        # 断言 sys_code 应为 "dbgpt"，表示系统代码为 dbgpt

    # 使用 dao 对象获取满足条件的用户列表的部分页结果
    res_half = dao.get_list_page({"user_name": "zhangsan"}, page=2, page_size=8)
    # 断言结果集的总数应为 10
    assert res_half.total_count == 10
    # 断言结果集的总页数应为 2
    assert res_half.total_pages == 2
    # 断言结果集当前页数为 2
    assert res_half.page == 2
    # 断言结果集每页大小为 8
    assert res_half.page_size == 8
    # 断言结果集中元素数量应为 2
    assert len(res_half.items) == 2
    
    # 遍历 res_half 对象的每个元素及其索引
    for i, r in enumerate(res_half.items):
        # 断言 chat_scene 应为 "chat_data"，表示聊天场景为数据聊天
        assert r.chat_scene == "chat_data"
        # 断言 sub_chat_scene 应为 "excel"，表示详细的聊天子场景为 Excel
        assert r.sub_chat_scene == "excel"
        # 断言 prompt_type 应为 "common"，表示提示类型为普通
        assert r.prompt_type == "common"
        # 断言 content 应为固定内容的字符串，提示编写 Python 中的快速排序函数
        assert r.content == "Write a qsort function in python."
        # 断言 user_name 应为 "zhangsan"，表示特定用户
        assert r.user_name == "zhangsan"
        # 断言 sys_code 应为 "dbgpt"，表示系统代码为 dbgpt
        assert r.sys_code == "dbgpt"
```