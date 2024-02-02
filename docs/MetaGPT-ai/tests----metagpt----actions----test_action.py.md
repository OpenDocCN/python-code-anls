# `MetaGPT\tests\metagpt\actions\test_action.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:43
@Author  : alexanderwu
@File    : test_action.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.actions 模块中导入 Action, ActionType, WritePRD, WriteTest 类
from metagpt.actions import Action, ActionType, WritePRD, WriteTest

# 测试 Action 类的 __repr__ 方法
def test_action_repr():
    # 创建 Action, WriteTest, WritePRD 实例
    actions = [Action(), WriteTest(), WritePRD()]
    # 断言 WriteTest 在 actions 中
    assert "WriteTest" in str(actions)

# 测试 ActionType 枚举类
def test_action_type():
    # 断言 WRITE_PRD 的值等于 WritePRD 类
    assert ActionType.WRITE_PRD.value == WritePRD
    # 断言 WRITE_TEST 的值等于 WriteTest 类
    assert ActionType.WRITE_TEST.value == WriteTest
    # 断言 WRITE_PRD 的名称为 "WRITE_PRD"
    assert ActionType.WRITE_PRD.name == "WRITE_PRD"
    # 断言 WRITE_TEST 的名称为 "WRITE_TEST"
    assert ActionType.WRITE_TEST.name == "WRITE_TEST"

# 测试创建简单的 Action 实例
def test_simple_action():
    # 创建一个名为 "AlexSay"，指令为 "Express your opinion with emotion and don't repeat it" 的 Action 实例
    action = Action(name="AlexSay", instruction="Express your opinion with emotion and don't repeat it")
    # 断言 action 的名称为 "AlexSay"
    assert action.name == "AlexSay"
    # 断言 action 的指令为 "Express your opinion with emotion and don't repeat it"
    assert action.node.instruction == "Express your opinion with emotion and don't repeat it"

# 测试创建空的 Action 实例
def test_empty_action():
    # 创建一个空的 Action 实例
    action = Action()
    # 断言 action 的名称为 "Action"
    assert action.name == "Action"
    # 断言 action 的 node 为空
    assert not action.node

# 异步测试，测试空的 Action 实例抛出 NotImplementedError 异常
@pytest.mark.asyncio
async def test_empty_action_exception():
    # 创建一个空的 Action 实例
    action = Action()
    # 断言运行 action 时会抛出 NotImplementedError 异常
    with pytest.raises(NotImplementedError):
        await action.run()

```