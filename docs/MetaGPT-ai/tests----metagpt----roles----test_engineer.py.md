# `MetaGPT\tests\metagpt\roles\test_engineer.py`

```py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/12 10:14
@Author  : alexanderwu
@File    : test_engineer.py
@Modified By: mashenquan, 2023-11-1. In accordance with Chapter 2.2.1 and 2.2.2 of RFC 116, utilize the new message
        distribution feature for message handling.
"""
# 导入所需的模块
import json
from pathlib import Path
# 导入 pytest 模块
import pytest
# 导入自定义模块
from metagpt.actions import WriteCode, WriteTasks
from metagpt.config import CONFIG
from metagpt.const import (
    PRDS_FILE_REPO,
    REQUIREMENT_FILENAME,
    SYSTEM_DESIGN_FILE_REPO,
    TASK_FILE_REPO,
)
# 导入日志模块
from metagpt.logs import logger
# 导入工程师角色模块
from metagpt.roles.engineer import Engineer
# 导入模式模块
from metagpt.schema import CodingContext, Message
# 导入通用工具模块
from metagpt.utils.common import CodeParser, any_to_name, any_to_str, aread, awrite
# 导入文件仓库模块
from metagpt.utils.file_repository import FileRepository
# 导入 git 仓库模块
from metagpt.utils.git_repository import ChangeType
# 导入模拟模块
from tests.metagpt.roles.mock import STRS_FOR_PARSING, TASKS, MockMessages

# 标记为异步测试
@pytest.mark.asyncio
async def test_engineer():
    # Prerequisites
    rqno = "20231221155954.json"
    # 保存文件
    await FileRepository.save_file(REQUIREMENT_FILENAME, content=MockMessages.req.content)
    await FileRepository.save_file(rqno, relative_path=PRDS_FILE_REPO, content=MockMessages.prd.content)
    await FileRepository.save_file(
        rqno, relative_path=SYSTEM_DESIGN_FILE_REPO, content=MockMessages.system_design.content
    )
    await FileRepository.save_file(rqno, relative_path=TASK_FILE_REPO, content=MockMessages.json_tasks.content)

    # 创建工程师对象
    engineer = Engineer()
    # 运行工程师对象的方法
    rsp = await engineer.run(Message(content="", cause_by=WriteTasks))

    logger.info(rsp)
    # 断言
    assert rsp.cause_by == any_to_str(WriteCode)
    src_file_repo = CONFIG.git_repo.new_file_repository(CONFIG.src_workspace)
    assert src_file_repo.changed_files

# 测试解析字符串
def test_parse_str():
    for idx, i in enumerate(STRS_FOR_PARSING):
        text = CodeParser.parse_str(f"{idx + 1}", i)
        # logger.info(text)
        assert text == "a"

# 测试解析块
def test_parse_blocks():
    tasks = CodeParser.parse_blocks(TASKS)
    logger.info(tasks.keys())
    assert "Task list" in tasks.keys()

# 测试解析文件列表
def test_parse_file_list():
    tasks = CodeParser.parse_file_list("Task list", TASKS)
    logger.info(tasks)
    assert isinstance(tasks, list)

# 测试解析代码
def test_parse_code():
    code = CodeParser.parse_code("Task list", TASKS, lang="python")
    logger.info(code)
    assert isinstance(code, str)

# 测试待办事项
def test_todo():
    role = Engineer()
    assert role.todo == any_to_name(WriteCode)

# 标记为异步测试
@pytest.mark.asyncio
async def test_new_coding_context():
    # Prerequisites
    demo_path = Path(__file__).parent / "../../data/demo_project"
    deps = json.loads(await aread(demo_path / "dependencies.json"))
    dependency = await CONFIG.git_repo.get_dependency()
    for k, v in deps.items():
        await dependency.update(k, set(v))
    data = await aread(demo_path / "system_design.json")
    rqno = "20231221155954.json"
    await awrite(CONFIG.git_repo.workdir / SYSTEM_DESIGN_FILE_REPO / rqno, data)
    data = await aread(demo_path / "tasks.json")
    await awrite(CONFIG.git_repo.workdir / TASK_FILE_REPO / rqno, data)

    CONFIG.src_workspace = Path(CONFIG.git_repo.workdir) / "game_2048"
    src_file_repo = CONFIG.git_repo.new_file_repository(relative_path=CONFIG.src_workspace)
    task_file_repo = CONFIG.git_repo.new_file_repository(relative_path=TASK_FILE_REPO)
    design_file_repo = CONFIG.git_repo.new_file_repository(relative_path=SYSTEM_DESIGN_FILE_REPO)

    filename = "game.py"
    ctx_doc = await Engineer._new_coding_doc(
        filename=filename,
        src_file_repo=src_file_repo,
        task_file_repo=task_file_repo,
        design_file_repo=design_file_repo,
        dependency=dependency,
    )
    assert ctx_doc
    assert ctx_doc.filename == filename
    assert ctx_doc.content
    ctx = CodingContext.model_validate_json(ctx_doc.content)
    assert ctx.filename == filename
    assert ctx.design_doc
    assert ctx.design_doc.content
    assert ctx.task_doc
    assert ctx.task_doc.content
    assert ctx.code_doc

    CONFIG.git_repo.add_change({f"{TASK_FILE_REPO}/{rqno}": ChangeType.UNTRACTED})
    CONFIG.git_repo.commit("mock env")
    await src_file_repo.save(filename=filename, content="content")
    role = Engineer()
    assert not role.code_todos
    await role._new_code_actions()
    assert role.code_todos

# 执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```