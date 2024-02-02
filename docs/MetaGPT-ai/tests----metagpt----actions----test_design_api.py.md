# `MetaGPT\tests\metagpt\actions\test_design_api.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 19:26
@Author  : alexanderwu
@File    : test_design_api.py
@Modifiled By: mashenquan, 2023-12-6. According to RFC 135
"""
# 导入 pytest 模块
import pytest

# 导入需要测试的模块和类
from metagpt.actions.design_api import WriteDesign
from metagpt.const import PRDS_FILE_REPO
from metagpt.logs import logger
from metagpt.schema import Message
from metagpt.utils.file_repository import FileRepository
from tests.metagpt.actions.mock_markdown import PRD_SAMPLE

# 使用 pytest 的装饰器标记为异步测试
@pytest.mark.asyncio
async def test_design_api():
    # 定义测试输入
    inputs = ["我们需要一个音乐播放器，它应该有播放、暂停、上一曲、下一曲等功能。", PRD_SAMPLE]
    # 遍历测试输入
    for prd in inputs:
        # 保存测试输入到文件
        await FileRepository.save_file("new_prd.txt", content=prd, relative_path=PRDS_FILE_REPO)

        # 创建 WriteDesign 实例
        design_api = WriteDesign()

        # 运行 WriteDesign 实例的方法，传入消息内容
        result = await design_api.run(Message(content=prd, instruct_content=None))
        # 记录日志
        logger.info(result)

        # 断言结果为真
        assert result

```