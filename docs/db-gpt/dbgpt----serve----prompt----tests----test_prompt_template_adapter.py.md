# `.\DB-GPT-src\dbgpt\serve\prompt\tests\test_prompt_template_adapter.py`

```py
# 导入 pytest 模块，用于单元测试
import pytest

# 导入相关模块和类
from dbgpt.core.interface.prompt import PromptManager, PromptTemplate
from dbgpt.storage.metadata import db
from dbgpt.storage.metadata.db_storage import SQLAlchemyStorage
from dbgpt.util.serialization.json_serialization import JsonSerializer

# 导入自定义模块
from ..models.prompt_template_adapter import PromptTemplateAdapter, ServeEntity


# 定义 serializer fixture，返回一个 JsonSerializer 实例
@pytest.fixture
def serializer():
    return JsonSerializer()


# 定义 db_url fixture，返回一个用于测试的内存 SQLite 数据库 URL
@pytest.fixture
def db_url():
    """Use in-memory SQLite database for testing"""
    return "sqlite:///:memory:"


# 定义 db_manager fixture，初始化数据库并创建所有表结构，返回 db 对象
@pytest.fixture
def db_manager(db_url):
    db.init_db(db_url)
    db.create_all()
    return db


# 定义 storage_adapter fixture，返回一个 PromptTemplateAdapter 实例
@pytest.fixture
def storage_adapter():
    return PromptTemplateAdapter()


# 定义 storage fixture，实例化 SQLAlchemyStorage 对象，返回 storage 实例
@pytest.fixture
def storage(db_manager, serializer, storage_adapter):
    storage = SQLAlchemyStorage(
        db_manager,
        ServeEntity,
        storage_adapter,
        serializer,
    )
    return storage


# 定义 prompt_manager fixture，实例化 PromptManager 对象，返回 prompt_manager 实例
@pytest.fixture
def prompt_manager(storage):
    return PromptManager(storage)


# 定义单元测试函数 test_save，测试保存操作
def test_save(prompt_manager: PromptManager):
    # 创建一个 PromptTemplate 实例
    prompt_template = PromptTemplate(
        template="hello {input}",
        input_variables=["input"],
        template_scene="chat_normal",
    )
    # 调用 prompt_manager 的 save 方法保存 prompt_template
    prompt_manager.save(
        prompt_template,
        prompt_name="hello",
    )

    # 使用 db.session() 进行数据库操作
    with db.session() as session:
        # 查询数据库中的数据
        result = (
            session.query(ServeEntity).filter(ServeEntity.prompt_name == "hello").all()
        )
        # 断言查询结果符合预期
        assert len(result) == 1
        assert result[0].prompt_name == "hello"
        assert result[0].content == "hello {input}"
        assert result[0].input_variables == "input"
    
    # 再次使用 db.session() 进行数据库操作
    with db.session() as session:
        # 断言数据库中 ServeEntity 表的记录数为 1
        assert session.query(ServeEntity).count() == 1
        # 断言查询不存在的 prompt_name 返回记录数为 0
        assert (
            session.query(ServeEntity)
            .filter(ServeEntity.prompt_name == "not exist prompt name")
            .count()
            == 0
        )


# 定义单元测试函数 test_prefer_query_language，测试按语言偏好查询操作
def test_prefer_query_language(prompt_manager: PromptManager):
    # 循环测试不同语言的 prompt_template
    for language in ["en", "zh"]:
        prompt_template = PromptTemplate(
            template="test",
            input_variables=[],
            template_scene="chat_normal",
        )
        # 调用 prompt_manager 的 save 方法保存 prompt_template
        prompt_manager.save(
            prompt_template,
            prompt_name="test_prompt",
            prompt_language=language,
        )
    
    # 测试偏好查询，优先选择中文，存在中文版本则返回中文 prompt_template
    result = prompt_manager.prefer_query("test_prompt", prefer_prompt_language="zh")
    assert len(result) == 1
    assert result[0].content == "test"
    assert result[0].prompt_language == "zh"
    
    # 测试偏好查询，选择不存在的语言，返回所有同名的 prompt_template
    result = prompt_manager.prefer_query(
        "test_prompt", prefer_prompt_language="not_exist"
    )
    assert len(result) == 2


# 定义单元测试函数 test_prefer_query_model，测试按模型偏好查询操作
def test_prefer_query_model(prompt_manager: PromptManager):
    # 这个测试函数目前是空的，用于后续添加对按模型偏好查询操作的测试
    pass
    # 遍历模型列表，对每个模型执行以下操作：
    for model in ["model1", "model2"]:
        # 创建一个新的提示模板对象，设定模板名称、输入变量为空列表，场景为普通聊天
        prompt_template = PromptTemplate(
            template="test",
            input_variables=[],
            template_scene="chat_normal",
        )
        # 将创建的提示模板保存到提示管理器中，指定模板名称为"test_prompt"，模型为当前迭代的模型
        prompt_manager.save(
            prompt_template,
            prompt_name="test_prompt",
            model=model,
        )

    # 使用prefer_model参数来查询指定名称的提示模板：
    # 如果prefer_model是"model1"且存在"model1"模型，则返回该模型的提示模板
    result = prompt_manager.prefer_query("test_prompt", prefer_model="model1")
    # 断言返回的结果长度为1
    assert len(result) == 1
    # 断言返回的第一个结果的内容为"test"
    assert result[0].content == "test"
    # 断言返回的第一个结果的模型为"model1"
    assert result[0].model == "model1"

    # 如果prefer_model是"not_exist"或者不存在该模型，则返回该名称下的所有提示模板
    result = prompt_manager.prefer_query("test_prompt", prefer_model="not_exist")
    # 断言返回的结果长度为2
    assert len(result) == 2
# 定义一个名为 test_list 的函数，接收一个类型为 PromptManager 的参数 prompt_manager
def test_list(prompt_manager: PromptManager):
    # 循环执行10次，i 的取值范围为 0 到 9
    for i in range(10):
        # 创建一个 PromptTemplate 对象，设定模板名称为 "test"，输入变量为空列表，场景为 "chat_normal"
        prompt_template = PromptTemplate(
            template="test",
            input_variables=[],
            template_scene="chat_normal",
        )
        # 将创建的 prompt_template 对象保存到 prompt_manager 中，设置 prompt_name 为 "test_prompt_{i}"，sys_code 根据 i 的奇偶性为 "dbgpt" 或 "not_dbgpt"
        prompt_manager.save(
            prompt_template,
            prompt_name=f"test_prompt_{i}",
            sys_code="dbgpt" if i % 2 == 0 else "not_dbgpt",
        )
    # 断言调用 prompt_manager 的 list 方法返回的结果长度为 10
    result = prompt_manager.list()
    assert len(result) == 10

    # 再次循环执行10次，i 的取值范围为 0 到 9
    for i in range(10):
        # 断言调用 prompt_manager 的 list 方法，并指定 prompt_name 参数为 f"test_prompt_{i}"，返回的结果长度为 1
        assert len(prompt_manager.list(prompt_name=f"test_prompt_{i}")) == 1
    # 断言调用 prompt_manager 的 list 方法，并指定 sys_code 参数为 "dbgpt"，返回的结果长度为 5
    assert len(prompt_manager.list(sys_code="dbgpt")) == 5
```