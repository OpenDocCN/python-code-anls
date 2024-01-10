# `MetaGPT\tests\metagpt\actions\test_write_prd.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : test_write_prd.py
@Modified By: mashenquan, 2023-11-1. According to Chapter 2.2.1 and 2.2.2 of RFC 116, replace `handle` with `run`.
"""
# 导入 pytest 模块
import pytest

# 导入需要测试的模块和类
from metagpt.actions import UserRequirement, WritePRD
from metagpt.config import CONFIG
from metagpt.const import DOCS_FILE_REPO, PRDS_FILE_REPO, REQUIREMENT_FILENAME
from metagpt.logs import logger
from metagpt.roles.product_manager import ProductManager
from metagpt.roles.role import RoleReactMode
from metagpt.schema import Message
from metagpt.utils.common import any_to_str
from metagpt.utils.file_repository import FileRepository

# 使用 pytest 的异步测试装饰器
@pytest.mark.asyncio
async def test_write_prd(new_filename):
    # 创建产品经理对象
    product_manager = ProductManager()
    # 设置需求内容
    requirements = "开发一个基于大语言模型与私有知识库的搜索引擎，希望可以基于大语言模型进行搜索总结"
    # 保存需求内容到文件
    await FileRepository.save_file(filename=REQUIREMENT_FILENAME, content=requirements, relative_path=DOCS_FILE_REPO)
    # 设置产品经理的反应模式
    product_manager.rc.react_mode = RoleReactMode.BY_ORDER
    # 运行产品经理的功能，生成产品需求文档
    prd = await product_manager.run(Message(content=requirements, cause_by=UserRequirement))
    # 断言产品需求文档的生成原因
    assert prd.cause_by == any_to_str(WritePRD)
    # 记录需求内容和产品需求文档
    logger.info(requirements)
    logger.info(prd)

    # 断言产品需求文档不为空
    assert prd is not None
    assert prd.content != ""
    # 断言文件仓库中有变更文件
    assert CONFIG.git_repo.new_file_repository(relative_path=PRDS_FILE_REPO).changed_files

# 如果是主程序入口
if __name__ == "__main__":
    # 运行 pytest 测试，并输出结果
    pytest.main([__file__, "-s"])

```