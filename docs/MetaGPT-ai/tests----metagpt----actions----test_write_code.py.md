# `MetaGPT\tests\metagpt\actions\test_write_code.py`

```

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 17:45
@Author  : alexanderwu
@File    : test_write_code.py
@Modifiled By: mashenquan, 2023-12-6. According to RFC 135
"""

# 导入所需的模块和库
from pathlib import Path
import pytest
from metagpt.actions.write_code import WriteCode
from metagpt.config import CONFIG
from metagpt.const import (
    CODE_SUMMARIES_FILE_REPO,
    SYSTEM_DESIGN_FILE_REPO,
    TASK_FILE_REPO,
    TEST_OUTPUTS_FILE_REPO,
)
from metagpt.logs import logger
from metagpt.provider.openai_api import OpenAILLM as LLM
from metagpt.schema import CodingContext, Document
from metagpt.utils.common import aread
from metagpt.utils.file_repository import FileRepository
from tests.metagpt.actions.mock_markdown import TASKS_2, WRITE_CODE_PROMPT_SAMPLE

# 异步测试函数，测试写代码的功能
@pytest.mark.asyncio
async def test_write_code():
    # 创建一个编码上下文
    context = CodingContext(
        filename="task_filename.py", design_doc=Document(content="设计一个名为'add'的函数，该函数接受两个整数作为输入，并返回它们的和。")
    )
    doc = Document(content=context.model_dump_json())
    write_code = WriteCode(context=doc)

    # 运行写代码的功能
    code = await write_code.run()
    logger.info(code.model_dump_json())

    # 检查生成的代码是否包含关键字
    assert "def add" in code.code_doc.content
    assert "return" in code.code_doc.content

# 异步测试函数，直接测试写代码的功能
@pytest.mark.asyncio
async def test_write_code_directly():
    prompt = WRITE_CODE_PROMPT_SAMPLE + "\n" + TASKS_2[0]
    llm = LLM()
    rsp = await llm.aask(prompt)
    logger.info(rsp)

# 异步测试函数，测试写代码的依赖
@pytest.mark.asyncio
async def test_write_code_deps():
    # 设置先决条件
    CONFIG.src_workspace = CONFIG.git_repo.workdir / "snake1/snake1"
    demo_path = Path(__file__).parent / "../../data/demo_project"
    await FileRepository.save_file(
        filename="test_game.py.json",
        content=await aread(str(demo_path / "test_game.py.json")),
        relative_path=TEST_OUTPUTS_FILE_REPO,
    )
    await FileRepository.save_file(
        filename="20231221155954.json",
        content=await aread(str(demo_path / "code_summaries.json")),
        relative_path=CODE_SUMMARIES_FILE_REPO,
    )
    await FileRepository.save_file(
        filename="20231221155954.json",
        content=await aread(str(demo_path / "system_design.json")),
        relative_path=SYSTEM_DESIGN_FILE_REPO,
    )
    await FileRepository.save_file(
        filename="20231221155954.json", content=await aread(str(demo_path / "tasks.json")), relative_path=TASK_FILE_REPO
    )
    await FileRepository.save_file(
        filename="main.py", content='if __name__ == "__main__":\nmain()', relative_path=CONFIG.src_workspace
    )
    context = CodingContext(
        filename="game.py",
        design_doc=await FileRepository.get_file(filename="20231221155954.json", relative_path=SYSTEM_DESIGN_FILE_REPO),
        task_doc=await FileRepository.get_file(filename="20231221155954.json", relative_path=TASK_FILE_REPO),
        code_doc=Document(filename="game.py", content="", root_path="snake1"),
    )
    coding_doc = Document(root_path="snake1", filename="game.py", content=context.json())

    action = WriteCode(context=coding_doc)
    rsp = await action.run()
    assert rsp
    assert rsp.code_doc.content

# 如果是主程序，则执行测试
if __name__ == "__main__":
    pytest.main([__file__, "-s"])

```