# `MetaGPT\tests\metagpt\actions\test_project_management.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:12
@Author  : alexanderwu
@File    : test_project_management.py
"""

# 导入 pytest 模块
import pytest

# 导入需要测试的模块和类
from metagpt.actions.project_management import WriteTasks
from metagpt.config import CONFIG
from metagpt.const import PRDS_FILE_REPO, SYSTEM_DESIGN_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.utils.file_repository import FileRepository
from tests.metagpt.actions.mock_json import DESIGN, PRD

# 标记为异步测试
@pytest.mark.asyncio
async def test_design_api():
    # 保存文件到指定的文件仓库
    await FileRepository.save_file("1.txt", content=str(PRD), relative_path=PRDS_FILE_REPO)
    await FileRepository.save_file("1.txt", content=str(DESIGN), relative_path=SYSTEM_DESIGN_FILE_REPO)
    # 打印 git 仓库配置信息
    logger.info(CONFIG.git_repo)

    # 创建 WriteTasks 实例
    action = WriteTasks()

    # 运行 WriteTasks 实例的 run 方法
    result = await action.run(Message(content="", instruct_content=None))
    # 打印运行结果
    logger.info(result)

    # 断言结果为真
    assert result

```