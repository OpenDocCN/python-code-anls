# `MetaGPT\tests\metagpt\serialize_deserialize\test_write_code.py`

```py

# 设置文件编码和作者信息
# 导入 pytest 模块
# 从 metagpt.actions 模块中导入 WriteCode 类
# 从 metagpt.schema 模块中导入 CodingContext 和 Document 类

# 定义测试函数 test_write_design_serialize
def test_write_design_serialize():
    # 创建 WriteCode 实例
    action = WriteCode()
    # 序列化 WriteCode 实例
    ser_action_dict = action.model_dump()
    # 断言序列化后的字典中包含指定的键值对
    assert ser_action_dict["name"] == "WriteCode"
    # 断言序列化后的字典中不包含指定的键
    assert "llm" not in ser_action_dict  # not export

# 标记异步测试函数
@pytest.mark.asyncio
# 定义测试函数 test_write_code_deserialize
async def test_write_code_deserialize():
    # 创建 CodingContext 实例
    context = CodingContext(
        filename="test_code.py", design_doc=Document(content="write add function to calculate two numbers")
    )
    # 创建 Document 实例
    doc = Document(content=context.model_dump_json())
    # 创建 WriteCode 实例
    action = WriteCode(context=doc)
    # 序列化 WriteCode 实例
    serialized_data = action.model_dump()
    # 创建新的 WriteCode 实例
    new_action = WriteCode(**serialized_data)

    # 断言新的 WriteCode 实例的名称
    assert new_action.name == "WriteCode"
    # 运行 WriteCode 实例的异步方法
    await action.run()

```