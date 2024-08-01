# `.\DB-GPT-src\dbgpt\core\interface\tests\test_prompt.py`

```py
# 导入json模块，用于处理JSON格式的数据
import json

# 导入pytest模块，用于编写和运行测试用例
import pytest

# 导入具体的类和函数，用于处理提示模板相关的逻辑
from dbgpt.core.interface.prompt import (
    PromptManager,
    PromptTemplate,
    StoragePromptTemplate,
)

# 导入查询规范类，用于定义数据库查询的具体规范
from dbgpt.core.interface.storage import QuerySpec

# 导入内存存储的fixture，用于在测试中提供内存级别的存储支持
from dbgpt.core.interface.tests.conftest import in_memory_storage


# 定义fixture，返回一个简单的存储提示模板对象
@pytest.fixture
def sample_storage_prompt_template():
    return StoragePromptTemplate(
        prompt_name="test_prompt",
        content="Sample content, {var1}, {var2}",
        prompt_language="en",
        prompt_format="f-string",
        input_variables="var1,var2",
        model="model1",
        chat_scene="scene1",
        sub_chat_scene="subscene1",
        prompt_type="type1",
        user_name="user1",
        sys_code="code1",
    )


# 定义fixture，返回一个复杂的存储提示模板对象
@pytest.fixture
def complex_storage_prompt_template():
    content = """Database name: {db_name} Table structure definition: {table_info} User Question:{user_input}"""
    return StoragePromptTemplate(
        prompt_name="chat_data_auto_execute_prompt",
        content=content,
        prompt_language="en",
        prompt_format="f-string",
        input_variables="db_name,table_info,user_input",
        model="vicuna-13b-v1.5",
        chat_scene="chat_data",
        sub_chat_scene="subscene1",
        prompt_type="common",
        user_name="zhangsan",
        sys_code="dbgpt",
    )


# 定义fixture，返回一个PromptManager对象，依赖于内存存储fixture
@pytest.fixture
def prompt_manager(in_memory_storage):
    return PromptManager(storage=in_memory_storage)


# 测试类：测试PromptTemplate类的功能
class TestPromptTemplate:
    
    # 测试方法：测试以f-string格式格式化模板字符串的功能
    @pytest.mark.parametrize(
        "template_str, input_vars, expected_output",
        [
            ("Hello {name}", {"name": "World"}, "Hello World"),
            ("{greeting}, {name}", {"greeting": "Hi", "name": "Alice"}, "Hi, Alice"),
        ],
    )
    def test_format_f_string(self, template_str, input_vars, expected_output):
        prompt = PromptTemplate(
            template=template_str,
            input_variables=list(input_vars.keys()),  # 提供输入变量的列表
            template_format="f-string",  # 指定模板格式为f-string
        )
        formatted_output = prompt.format(**input_vars)  # 格式化模板
        assert formatted_output == expected_output  # 断言格式化结果与期望输出一致

    # 测试方法：测试以Jinja2格式格式化模板字符串的功能
    @pytest.mark.parametrize(
        "template_str, input_vars, expected_output",
        [
            ("Hello {{ name }}", {"name": "World"}, "Hello World"),
            (
                "{{ greeting }}, {{ name }}",
                {"greeting": "Hi", "name": "Alice"},
                "Hi, Alice",
            ),
        ],
    )
    def test_format_jinja2(self, template_str, input_vars, expected_output):
        prompt = PromptTemplate(
            template=template_str,
            input_variables=list(input_vars.keys()),  # 提供输入变量的列表
            template_format="jinja2",  # 指定模板格式为Jinja2
        )
        formatted_output = prompt.format(**input_vars)  # 格式化模板
        assert formatted_output == expected_output  # 断言格式化结果与期望输出一致
    # 测试带有响应格式的模板字符串格式化方法
    def test_format_with_response_format(self):
        # 定义模板字符串
        template_str = "Response: {response}"
        # 创建模板对象，指定输入变量和模板格式为 f-string，响应格式为 JSON 序列化后的消息 "hello"
        prompt = PromptTemplate(
            template=template_str,
            input_variables=["response"],
            template_format="f-string",
            response_format=json.dumps({"message": "hello"}),
        )
        # 格式化模板，传入 response="hello"，生成格式化后的输出
        formatted_output = prompt.format(response="hello")
        # 断言格式化后的输出包含 "Response: "
        assert "Response: " in formatted_output

    # 测试缺少变量时的模板字符串格式化方法
    def test_format_missing_variable(self):
        # 定义模板字符串
        template_str = "Hello {name}"
        # 创建模板对象，指定输入变量和模板格式为 f-string
        prompt = PromptTemplate(
            template=template_str, input_variables=["name"], template_format="f-string"
        )
        # 使用 pytest 检查缺少变量时是否会引发 KeyError 异常
        with pytest.raises(KeyError):
            prompt.format()

    # 测试额外变量时的模板字符串格式化方法（宽松模式）
    def test_format_extra_variable(self):
        # 定义模板字符串
        template_str = "Hello {name}"
        # 创建模板对象，指定输入变量和模板格式为 f-string，并设置宽松模式
        prompt = PromptTemplate(
            template_str,
            input_variables=["name"],
            template_format="f-string",
            template_is_strict=False,
        )
        # 格式化模板，传入 name="World" 和额外未使用的变量 "extra"
        formatted_output = prompt.format(name="World", extra="unused")
        # 断言格式化后的输出等于 "Hello World"
        assert formatted_output == "Hello World"

    # 测试复杂情况下的模板字符串格式化方法
    def test_format_complex(self, complex_storage_prompt_template):
        # 将复杂存储模板转换为普通模板对象
        prompt = complex_storage_prompt_template.to_prompt_template()
        # 格式化模板，传入多个变量
        formatted_output = prompt.format(
            db_name="db1",
            table_info="create table users(id int, name varchar(20))",
            user_input="find all users whose name is 'Alice'",
        )
        # 断言格式化后的输出符合预期的复杂文本
        assert (
            formatted_output
            == "Database name: db1 Table structure definition: create table users(id int, name varchar(20)) "
            "User Question:find all users whose name is 'Alice'"
        )
# 定义一个测试类 TestStoragePromptTemplate
class TestStoragePromptTemplate:
    # 测试构造函数和属性
    def test_constructor_and_properties(self):
        # 创建一个 StoragePromptTemplate 实例
        storage_item = StoragePromptTemplate(
            prompt_name="test",
            content="Hello {name}",
            prompt_language="en",
            prompt_format="f-string",
            input_variables="name",
            model="model1",
            chat_scene="chat",
            sub_chat_scene="sub_chat",
            prompt_type="type",
            user_name="user",
            sys_code="sys",
        )
        # 断言各属性值是否正确
        assert storage_item.prompt_name == "test"
        assert storage_item.content == "Hello {name}"
        assert storage_item.prompt_language == "en"
        assert storage_item.prompt_format == "f-string"
        assert storage_item.input_variables == "name"
        assert storage_item.model == "model1"

    # 测试构造函数异常情况
    def test_constructor_exceptions(self):
        # 断言构造函数是否会抛出 ValueError 异常
        with pytest.raises(ValueError):
            StoragePromptTemplate(prompt_name=None, content="Hello")

    # 测试转换为 PromptTemplate 对象
    def test_to_prompt_template(self, sample_storage_prompt_template):
        # 转换为 PromptTemplate 对象
        prompt_template = sample_storage_prompt_template.to_prompt_template()
        # 断言转换结果是否正确
        assert isinstance(prompt_template, PromptTemplate)
        assert prompt_template.template == "Sample content, {var1}, {var2}"
        assert prompt_template.input_variables == ["var1", "var2"]

    # 测试从 PromptTemplate 对象创建 StoragePromptTemplate 对象
    def test_from_prompt_template(self):
        # 创建一个 PromptTemplate 对象
        prompt_template = PromptTemplate(
            template="Sample content, {var1}, {var2}",
            input_variables=["var1", "var2"],
            template_format="f-string",
        )
        # 从 PromptTemplate 对象创建 StoragePromptTemplate 对象
        storage_prompt_template = StoragePromptTemplate.from_prompt_template(
            prompt_template=prompt_template, prompt_name="test_prompt"
        )
        # 断言创建结果是否正确
        assert storage_prompt_template.prompt_name == "test_prompt"
        assert storage_prompt_template.content == "Sample content, {var1}, {var2}"
        assert storage_prompt_template.input_variables == "var1,var2"

    # 测试合并两个 StoragePromptTemplate 对象
    def test_merge(self, sample_storage_prompt_template):
        # 创建另一个 StoragePromptTemplate 对象
        other = StoragePromptTemplate(
            prompt_name="other_prompt",
            content="Other content",
        )
        # 合并两个对象
        sample_storage_prompt_template.merge(other)
        # 断言合并结果是否正确
        assert sample_storage_prompt_template.content == "Other content"

    # 测试将 StoragePromptTemplate 对象转换为字典
    def test_to_dict(self, sample_storage_prompt_template):
        # 转换为字典
        result = sample_storage_prompt_template.to_dict()
        # 断言转换结果是否正确
        assert result == {
            "prompt_name": "test_prompt",
            "content": "Sample content, {var1}, {var2}",
            "prompt_language": "en",
            "prompt_format": "f-string",
            "input_variables": "var1,var2",
            "model": "model1",
            "chat_scene": "scene1",
            "sub_chat_scene": "subscene1",
            "prompt_type": "type1",
            "user_name": "user1",
            "sys_code": "code1",
        }

    # 测试保存和加载 StoragePromptTemplate 对象
    def test_save_and_load_storage(
        self, sample_storage_prompt_template, in_memory_storage
    ):
        存储 sample_storage_prompt_template 到 in_memory_storage
        从 in_memory_storage 中加载 sample_storage_prompt_template，并赋给 loaded_item
        断言 loaded_item 的 content 属性应为 "Sample content, {var1}, {var2}"

    def test_check_exceptions(self):
        使用 pytest 来检查是否会抛出 ValueError 异常
        调用 StoragePromptTemplate 类，传入 prompt_name=None 和 content="Hello" 来触发异常

    def test_from_object(self, sample_storage_prompt_template):
        创建一个名为 other 的 StoragePromptTemplate 实例，prompt_name 设置为 "other"，content 设置为 "Other content"
        使用 sample_storage_prompt_template 的 from_object 方法，将 other 对象的属性复制给 sample_storage_prompt_template
        断言 sample_storage_prompt_template 的 content 属性应为 "Other content"
        断言 sample_storage_prompt_template 的 input_variables 不应为 "var1,var2"
        断言 sample_storage_prompt_template 的 prompt_name 属性不应被改变，仍为 "test_prompt"
        断言 sample_storage_prompt_template 的 sys_code 属性应为 "code1"
class TestPromptManager:
    # 测试保存方法
    def test_save(self, prompt_manager, in_memory_storage):
        # 创建一个提示模板对象
        prompt_template = PromptTemplate(
            template="hello {input}",  # 设置模板内容
            input_variables=["input"],  # 设置输入变量列表
            template_scene="chat_normal",  # 设置模板场景
        )
        # 调用 prompt_manager 的保存方法，保存提示模板
        prompt_manager.save(
            prompt_template,  # 传入要保存的提示模板对象
            prompt_name="hello",  # 指定保存的提示名称
        )
        # 查询内存存储，检查是否保存成功
        result = in_memory_storage.query(
            QuerySpec(conditions={"prompt_name": "hello"}), StoragePromptTemplate
        )
        # 断言查询结果长度为1
        assert len(result) == 1
        # 断言查询结果的内容与预期一致
        assert result[0].content == "hello {input}"

    # 测试简单查询方法
    def test_prefer_query_simple(self, prompt_manager, in_memory_storage):
        # 在内存存储中保存一个简单的提示模板
        in_memory_storage.save(
            StoragePromptTemplate(prompt_name="test_prompt", content="test")
        )
        # 调用 prompt_manager 的查询方法，查询指定名称的提示模板
        result = prompt_manager.prefer_query("test_prompt")
        # 断言查询结果长度为1
        assert len(result) == 1
        # 断言查询结果的内容与预期一致
        assert result[0].content == "test"

    # 测试语言偏好查询方法
    def test_prefer_query_language(self, prompt_manager, in_memory_storage):
        # 分别以英文和中文保存同名但语言不同的提示模板
        for language in ["en", "zh"]:
            in_memory_storage.save(
                StoragePromptTemplate(
                    prompt_name="test_prompt",
                    content="test",
                    prompt_language=language,
                )
            )
        # 偏好中文，如果存在中文提示模板，则返回中文的提示模板
        result = prompt_manager.prefer_query("test_prompt", prefer_prompt_language="zh")
        assert len(result) == 1
        assert result[0].content == "test"
        assert result[0].prompt_language == "zh"
        # 偏好语言不存在，将返回所有同名的提示模板
        result = prompt_manager.prefer_query(
            "test_prompt", prefer_prompt_language="not_exist"
        )
        assert len(result) == 2

    # 测试模型偏好查询方法
    def test_prefer_query_model(self, prompt_manager, in_memory_storage):
        # 分别以 model1 和 model2 保存同名但模型不同的提示模板
        for model in ["model1", "model2"]:
            in_memory_storage.save(
                StoragePromptTemplate(
                    prompt_name="test_prompt", content="test", model=model
                )
            )
        # 偏好 model1，如果存在 model1 的提示模板，则返回 model1 的提示模板
        result = prompt_manager.prefer_query("test_prompt", prefer_model="model1")
        assert len(result) == 1
        assert result[0].content == "test"
        assert result[0].model == "model1"
        # 偏好模型不存在，将返回所有同名的提示模板
        result = prompt_manager.prefer_query("test_prompt", prefer_model="not_exist")
        assert len(result) == 2
    # 定义一个测试方法，用于测试 PromptManager 的列表功能和删除功能
    def test_list(self, prompt_manager, in_memory_storage):
        # 保存一个模板到 PromptManager 中，模板内容是 "Hello {name}"，包含一个输入变量 "name"
        prompt_manager.save(
            PromptTemplate(template="Hello {name}", input_variables=["name"]),
            prompt_name="name1",  # 设置模板的名称为 "name1"
        )
        # 保存另一个 SQL 模板到 PromptManager 中，模板内容包含输入变量 "dialect" 和 "table_name"
        prompt_manager.save(
            PromptTemplate(
                template="Write a SQL of {dialect} to query all data of {table_name}.",
                input_variables=["dialect", "table_name"],
            ),
            prompt_name="sql_template",  # 设置模板的名称为 "sql_template"
        )
        # 获取 PromptManager 中的所有模板列表
        all_templates = prompt_manager.list()
        # 断言：所有模板数量应该为 2
        assert len(all_templates) == 2
        # 断言：模板名称为 "name1" 的模板数量应该为 1
        assert len(prompt_manager.list(prompt_name="name1")) == 1
        # 断言：模板名称为 "not exist" 的模板数量应该为 0
        assert len(prompt_manager.list(prompt_name="not exist")) == 0

    # 定义一个测试方法，用于测试 PromptManager 的删除功能
    def test_delete(self, prompt_manager, in_memory_storage):
        # 保存一个模板到 PromptManager 中，模板内容是 "Hello {name}"，包含一个输入变量 "name"
        prompt_manager.save(
            PromptTemplate(template="Hello {name}", input_variables=["name"]),
            prompt_name="to_delete",  # 设置模板的名称为 "to_delete"
        )
        # 删除名称为 "to_delete" 的模板
        prompt_manager.delete("to_delete")
        # 查询内存存储中所有条件为 {"prompt_name": "to_delete"} 的存储对象，并返回结果
        result = in_memory_storage.query(
            QuerySpec(conditions={"prompt_name": "to_delete"}), StoragePromptTemplate
        )
        # 断言：查询结果中应该没有符合条件的对象，即长度应该为 0
        assert len(result) == 0
```