# `MetaGPT\tests\metagpt\serialize_deserialize\test_sk_agent.py`

```

# 设置文件编码为 utf-8
# 导入 pytest 模块
import pytest

# 从 metagpt.roles.sk_agent 模块中导入 SkAgent 类

# 测试 SkAgent 类的序列化方法
def test_sk_agent_serialize():
    # 创建 SkAgent 对象
    role = SkAgent()
    # 调用 model_dump 方法进行序列化，排除指定字段
    ser_role_dict = role.model_dump(exclude={"import_semantic_skill_from_directory", "import_skill"})
    # 断言结果字典中包含 "name" 键
    assert "name" in ser_role_dict
    # 断言结果字典中包含 "planner" 键

# 异步测试 SkAgent 类的反序列化方法
@pytest.mark.asyncio
async def test_sk_agent_deserialize():
    # 创建 SkAgent 对象
    role = SkAgent()
    # 调用 model_dump 方法进行序列化，排除指定字段
    ser_role_dict = role.model_dump(exclude={"import_semantic_skill_from_directory", "import_skill"})
    # 断言结果字典中包含 "name" 键
    assert "name" in ser_role_dict
    # 断言结果字典中包含 "planner" 键

    # 根据序列化后的字典创建新的 SkAgent 对象
    new_role = SkAgent(**ser_role_dict)
    # 断言新对象的 name 属性为 "Sunshine"
    assert new_role.name == "Sunshine"
    # 断言新对象的 actions 列表长度为 1
    assert len(new_role.actions) == 1

```