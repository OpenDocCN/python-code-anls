# `MetaGPT\tests\metagpt\serialize_deserialize\test_tutorial_assistant.py`

```py

# 设置文件编码为 UTF-8
# 导入 pytest 模块

# 从 metagpt.actions.write_tutorial 模块中导入 WriteDirectory 类
# 从 metagpt.roles.tutorial_assistant 模块中导入 TutorialAssistant 类

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
async def test_tutorial_assistant_deserialize():
    # 创建 TutorialAssistant 实例
    role = TutorialAssistant()
    # 将 role 对象序列化为字典
    ser_role_dict = role.model_dump()
    # 断言字典中包含特定的键
    assert "name" in ser_role_dict
    assert "language" in ser_role_dict
    assert "topic" in ser_role_dict

    # 使用序列化后的字典创建新的 TutorialAssistant 实例
    new_role = TutorialAssistant(**ser_role_dict)
    # 断言新实例的属性值符合预期
    assert new_role.name == "Stitch"
    assert len(new_role.actions) == 1
    assert isinstance(new_role.actions[0], WriteDirectory)

```