# `MetaGPT\tests\metagpt\serialize_deserialize\test_write_design.py`

```py

# 设置文件编码和作者信息
# 导入 pytest 模块
import pytest
# 从 metagpt.actions 模块中导入 WriteDesign 和 WriteTasks 类

# 测试 WriteDesign 类的序列化方法
def test_write_design_serialize():
    # 创建 WriteDesign 实例
    action = WriteDesign()
    # 调用 model_dump 方法序列化实例
    ser_action_dict = action.model_dump()
    # 断言序列化后的字典中包含 "name" 键
    assert "name" in ser_action_dict
    # 断言序列化后的字典中不包含 "llm" 键（不导出）
    assert "llm" not in ser_action_dict  

# 测试 WriteTasks 类的序列化方法
def test_write_task_serialize():
    # 创建 WriteTasks 实例
    action = WriteTasks()
    # 调用 model_dump 方法序列化实例
    ser_action_dict = action.model_dump()
    # 断言序列化后的字典中包含 "name" 键
    assert "name" in ser_action_dict
    # 断言序列化后的字典中不包含 "llm" 键（不导出）
    assert "llm" not in ser_action_dict  

# 异步测试 WriteDesign 类的反序列化方法
@pytest.mark.asyncio
async def test_write_design_deserialize():
    # 创建 WriteDesign 实例
    action = WriteDesign()
    # 调用 model_dump 方法序列化实例
    serialized_data = action.model_dump()
    # 使用序列化数据创建新的 WriteDesign 实例
    new_action = WriteDesign(**serialized_data)
    # 断言新实例的 name 属性为 "WriteDesign"
    assert new_action.name == "WriteDesign"
    # 调用 run 方法并传入消息，测试异步运行
    await new_action.run(with_messages="write a cli snake game")

# 异步测试 WriteTasks 类的反序列化方法
@pytest.mark.asyncio
async def test_write_task_deserialize():
    # 创建 WriteTasks 实例
    action = WriteTasks()
    # 调用 model_dump 方法序列化实例
    serialized_data = action.model_dump()
    # 使用序列化数据创建新的 WriteTasks 实例
    new_action = WriteTasks(**serialized_data)
    # 断言新实例的 name 属性为 "WriteTasks"
    assert new_action.name == "WriteTasks"
    # 调用 run 方法并传入消息，测试异步运行
    await new_action.run(with_messages="write a cli snake game")

```