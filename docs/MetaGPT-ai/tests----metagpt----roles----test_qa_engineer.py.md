# `MetaGPT\tests\metagpt\roles\test_qa_engineer.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 12:01
@Author  : alexanderwu
@File    : test_qa_engineer.py
"""
# 导入模块
from pathlib import Path
from typing import List
# 导入 pytest 模块
import pytest
# 导入 pydantic 模块中的 Field 类
from pydantic import Field
# 导入自定义模块
from metagpt.actions import DebugError, RunCode, WriteTest
from metagpt.actions.summarize_code import SummarizeCode
from metagpt.config import CONFIG
from metagpt.environment import Environment
from metagpt.roles import QaEngineer
from metagpt.schema import Message
from metagpt.utils.common import any_to_str, aread, awrite

# 异步测试函数
async def test_qa():
    # 设置 demo_path 变量
    demo_path = Path(__file__).parent / "../../data/demo_project"
    # 设置 CONFIG.src_workspace 变量
    CONFIG.src_workspace = Path(CONFIG.git_repo.workdir) / "qa/game_2048"
    # 读取文件内容
    data = await aread(filename=demo_path / "game.py", encoding="utf-8")
    # 写入文件内容
    await awrite(filename=CONFIG.src_workspace / "game.py", data=data, encoding="utf-8")
    # 写入空文件
    await awrite(filename=Path(CONFIG.git_repo.workdir) / "requirements.txt", data="")

    # 定义 MockEnv 类
    class MockEnv(Environment):
        msgs: List[Message] = Field(default_factory=list)

        # 发布消息方法
        def publish_message(self, message: Message, peekable: bool = True) -> bool:
            self.msgs.append(message)
            return True

    # 实例化 MockEnv 类
    env = MockEnv()

    # 实例化 QaEngineer 角色
    role = QaEngineer()
    # 设置环境
    role.set_env(env)
    # 运行角色
    await role.run(with_message=Message(content="", cause_by=SummarizeCode))
    # 断言消息列表不为空
    assert env.msgs
    # 断言消息列表中第一个消息的 cause_by 属性为 WriteTest
    assert env.msgs[0].cause_by == any_to_str(WriteTest)
    # 获取消息
    msg = env.msgs[0]
    env.msgs.clear()
    # 运行角色
    await role.run(with_message=msg)
    # 断言消息列表不为空
    assert env.msgs
    # 断言消息列表中第一个消息的 cause_by 属性为 RunCode
    assert env.msgs[0].cause_by == any_to_str(RunCode)
    # 获取消息
    msg = env.msgs[0]
    env.msgs.clear()
    # 运行角色
    await role.run(with_message=msg)
    # 断言消息列表不为空
    assert env.msgs
    # 断言消息列表中第一个消息的 cause_by 属性为 DebugError
    assert env.msgs[0].cause_by == any_to_str(DebugError)
    # 获取消息
    msg = env.msgs[0]
    env.msgs.clear()
    # 设置测试轮次为 1
    role.test_round_allowed = 1
    # 运行角色
    rsp = await role.run(with_message=msg)
    # 断言响应内容中包含 "Exceeding"
    assert "Exceeding" in rsp.content

# 主函数入口
if __name__ == "__main__":
    # 运行 pytest 测试
    pytest.main([__file__, "-s"])

```