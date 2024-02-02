# `MetaGPT\tests\metagpt\roles\test_project_manager.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 10:23
@Author  : alexanderwu
@File    : test_project_manager.py
"""
# 导入 pytest 模块
import pytest

# 从 metagpt.logs 模块中导入 logger 对象
from metagpt.logs import logger
# 从 metagpt.roles 模块中导入 ProjectManager 类
from metagpt.roles import ProjectManager
# 从 tests.metagpt.roles.mock 模块中导入 MockMessages 对象
from tests.metagpt.roles.mock import MockMessages

# 使用 pytest.mark.asyncio 装饰器标记异步测试函数
@pytest.mark.asyncio
# 定义测试函数 test_project_manager
async def test_project_manager():
    # 创建 ProjectManager 对象
    project_manager = ProjectManager()
    # 调用 ProjectManager 对象的 run 方法，并传入 MockMessages.system_design 参数，获取返回结果
    rsp = await project_manager.run(MockMessages.system_design)
    # 记录返回结果
    logger.info(rsp)

```