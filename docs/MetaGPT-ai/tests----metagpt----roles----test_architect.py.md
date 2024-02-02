# `MetaGPT\tests\metagpt\roles\test_architect.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/20 14:37
@Author  : alexanderwu
@File    : test_architect.py
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.2.1 and 2.2.2 of RFC 116, utilize the new message
        distribution feature for message handling.
"""
# 导入所需的模块
import uuid
import pytest
# 导入自定义模块
from metagpt.actions import WriteDesign, WritePRD
from metagpt.config import CONFIG
from metagpt.const import PRDS_FILE_REPO
from metagpt.logs import logger
from metagpt.roles import Architect
from metagpt.schema import Message
from metagpt.utils.common import any_to_str, awrite
from tests.metagpt.roles.mock import MockMessages

# 使用 pytest 的异步测试装饰器
@pytest.mark.asyncio
async def test_architect():
    # Prerequisites
    # 生成一个随机的文件名
    filename = uuid.uuid4().hex + ".json"
    # 将模拟消息的内容写入指定的文件路径
    await awrite(CONFIG.git_repo.workdir / PRDS_FILE_REPO / filename, data=MockMessages.prd.content)

    # 创建 Architect 角色对象
    role = Architect()
    # 发送消息给 Architect 角色，并获取响应
    rsp = await role.run(with_message=Message(content="", cause_by=WritePRD))
    # 记录日志
    logger.info(rsp)
    # 断言响应内容长度大于0
    assert len(rsp.content) > 0
    # 断言响应的 cause_by 属性为 WriteDesign
    assert rsp.cause_by == any_to_str(WriteDesign)

    # test update
    # 发送更新消息给 Architect 角色，并获取响应
    rsp = await role.run(with_message=Message(content="", cause_by=WritePRD))
    # 断言响应存在
    assert rsp
    # 断言响应的 cause_by 属性为 WriteDesign
    assert rsp.cause_by == any_to_str(WriteDesign)
    # 断言响应内容长度大于0
    assert len(rsp.content) > 0

# 如果当前脚本为主程序，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```