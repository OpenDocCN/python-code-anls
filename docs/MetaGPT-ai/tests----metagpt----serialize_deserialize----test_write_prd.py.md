# `MetaGPT\tests\metagpt\serialize_deserialize\test_write_prd.py`

```py

# -*- coding: utf-8 -*-  # 设置文件编码格式为 UTF-8
# @Date    : 11/22/2023 1:47 PM  # 代码编写日期和时间
# @Author  : stellahong (stellahong@fuzhi.ai)  # 作者信息
# @Desc    : 代码描述

import pytest  # 导入 pytest 模块

from metagpt.actions import WritePRD  # 从 metagpt.actions 模块导入 WritePRD 类
from metagpt.schema import Message  # 从 metagpt.schema 模块导入 Message 类

# 测试序列化操作的函数
def test_action_serialize(new_filename):
    action = WritePRD()  # 创建 WritePRD 实例
    ser_action_dict = action.model_dump()  # 序列化 WritePRD 实例
    assert "name" in ser_action_dict  # 断言序列化后的字典中包含 "name" 键
    assert "llm" not in ser_action_dict  # 断言序列化后的字典中不包含 "llm" 键（不导出）

# 异步测试反序列化操作的函数
@pytest.mark.asyncio
async def test_action_deserialize(new_filename):
    action = WritePRD()  # 创建 WritePRD 实例
    serialized_data = action.model_dump()  # 序列化 WritePRD 实例
    new_action = WritePRD(**serialized_data)  # 使用序列化数据反序列化创建新的 WritePRD 实例
    assert new_action.name == "WritePRD"  # 断言新实例的 name 属性为 "WritePRD"
    action_output = await new_action.run(with_messages=Message(content="write a cli snake game"))  # 运行新实例的 run 方法，并传入 Message 实例
    assert len(action_output.content) > 0  # 断言运行结果的 content 属性长度大于 0

```