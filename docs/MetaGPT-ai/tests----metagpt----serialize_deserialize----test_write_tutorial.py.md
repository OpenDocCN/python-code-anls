# `MetaGPT\tests\metagpt\serialize_deserialize\test_write_tutorial.py`

```py

# -*- coding: utf-8 -*-
# @Desc    : 设置文件编码和描述信息

from typing import Dict  # 导入 Dict 类型提示

import pytest  # 导入 pytest 测试框架

from metagpt.actions.write_tutorial import WriteContent, WriteDirectory  # 导入自定义模块中的 WriteContent 和 WriteDirectory 类


@pytest.mark.asyncio  # 标记为异步测试
@pytest.mark.parametrize(("language", "topic"), [("English", "Write a tutorial about Python")])  # 参数化测试
async def test_write_directory_deserialize(language: str, topic: str):
    action = WriteDirectory()  # 创建 WriteDirectory 实例
    serialized_data = action.model_dump()  # 序列化实例数据
    assert serialized_data["name"] == "WriteDirectory"  # 断言实例数据中的名称为 "WriteDirectory"
    assert serialized_data["language"] == "Chinese"  # 断言实例数据中的语言为 "Chinese"

    new_action = WriteDirectory(**serialized_data)  # 使用序列化数据创建新的 WriteDirectory 实例
    ret = await new_action.run(topic=topic)  # 运行实例的 run 方法
    assert isinstance(ret, dict)  # 断言返回结果为字典类型
    assert "title" in ret  # 断言返回结果中包含 "title" 键
    assert "directory" in ret  # 断言返回结果中包含 "directory" 键
    assert isinstance(ret["directory"], list)  # 断言返回结果中 "directory" 值为列表类型
    assert len(ret["directory"])  # 断言返回结果中 "directory" 列表不为空
    assert isinstance(ret["directory"][0], dict)  # 断言返回结果中 "directory" 列表的第一个元素为字典类型


@pytest.mark.asyncio  # 标记为异步测试
@pytest.mark.parametrize(
    ("language", "topic", "directory"),
    [("English", "Write a tutorial about Python", {"Introduction": ["What is Python?", "Why learn Python?"]})],  # 参数化测试
)
async def test_write_content_deserialize(language: str, topic: str, directory: Dict):
    action = WriteContent(language=language, directory=directory)  # 创建 WriteContent 实例
    serialized_data = action.model_dump()  # 序列化实例数据
    assert serialized_data["name"] == "WriteContent"  # 断言实例数据中的名称为 "WriteContent"

    new_action = WriteContent(**serialized_data)  # 使用序列化数据创建新的 WriteContent 实例
    ret = await new_action.run(topic=topic)  # 运行实例的 run 方法
    assert isinstance(ret, str)  # 断言返回结果为字符串类型
    assert list(directory.keys())[0] in ret  # 断言返回结果中包含 directory 字典的第一个键
    for value in list(directory.values())[0]:  # 遍历 directory 字典的第一个值
        assert value in ret  # 断言值在返回结果中

```