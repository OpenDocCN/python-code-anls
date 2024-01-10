# `MetaGPT\tests\metagpt\serialize_deserialize\test_product_manager.py`

```

# -*- coding: utf-8 -*-  # 设置文件编码格式为 UTF-8
# @Date    : 11/26/2023 2:07 PM  # 代码编写日期和时间
# @Author  : stellahong (stellahong@fuzhi.ai)  # 作者信息
# @Desc    :  # 代码描述

import pytest  # 导入 pytest 模块

from metagpt.actions.action import Action  # 从 metagpt.actions.action 模块导入 Action 类
from metagpt.roles.product_manager import ProductManager  # 从 metagpt.roles.product_manager 模块导入 ProductManager 类
from metagpt.schema import Message  # 从 metagpt.schema 模块导入 Message 类

# 使用 pytest.mark.asyncio 装饰器标记为异步测试函数
@pytest.mark.asyncio
async def test_product_manager_deserialize(new_filename):
    # 创建 ProductManager 实例
    role = ProductManager()
    # 将 role 对象序列化为字典
    ser_role_dict = role.model_dump(by_alias=True)
    # 使用序列化后的字典创建新的 ProductManager 实例
    new_role = ProductManager(**ser_role_dict)

    # 断言新实例的 name 属性为 "Alice"
    assert new_role.name == "Alice"
    # 断言新实例的 actions 列表长度为 2
    assert len(new_role.actions) == 2
    # 断言新实例的 actions[0] 是 Action 类的实例
    assert isinstance(new_role.actions[0], Action)
    # 调用 actions[0] 的 run 方法，传入 Message 对象作为参数
    await new_role.actions[0].run([Message(content="write a cli snake game")])

```